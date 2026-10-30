"""
DataManagerV2: Modular orchestration layer for multi-omics data management.

This module provides the new DataManagerV2 class that orchestrates
modality adapters, storage backends, and validation to enable
flexible multi-omics data analysis with complete provenance tracking.
"""

import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import zipfile
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict, Union
from unittest.mock import Mock

import pandas as pd

from lobster.core.adapters.base import BaseAdapter
from lobster.core.adapters.factory import create_default_adapters
from lobster.core.exceptions import HdfReadRefusal
from lobster.core.export_scope import (
    assert_inside_workspace,
    check_modality_name,
    outside_workspace,
)
from lobster.core.figures.errors import ManagedPathError
from lobster.core.plot_manager import PlotManager
from lobster.core.provenance.analysis_ir import (
    create_adapter_loading_ir,
    create_data_loading_ir,
    create_data_saving_ir,
)

# Adapter classes stay importable from this module for existing callers;
# default adapters are constructed by lobster.core.adapters.factory.
# Import TranscriptomicsAdapter conditionally (has optional scanpy dependency)
try:
    from lobster.core.adapters.transcriptomics_adapter import TranscriptomicsAdapter

    TRANSCRIPTOMICS_ADAPTER_AVAILABLE = True
except ImportError:
    TRANSCRIPTOMICS_ADAPTER_AVAILABLE = False
    TranscriptomicsAdapter = None  # type: ignore

# Import H5AD backend conditionally (may have optional dependencies)
try:
    from lobster.core.backends.h5ad_backend import H5ADBackend

    H5AD_BACKEND_AVAILABLE = True
except ImportError:
    H5AD_BACKEND_AVAILABLE = False
    H5ADBackend = None  # type: ignore
from lobster.core.interfaces.adapter import IModalityAdapter
from lobster.core.interfaces.backend import IDataBackend
from lobster.core.interfaces.validator import ValidationResult
from lobster.core.provenance.provenance import ProvenanceTracker
from lobster.core.queues.queue_storage import atomic_write_json, queue_file_lock
from lobster.core.runtime.workspace import resolve_workspace
from lobster.core.session_identity import (
    display_client_session_id as _display_client_session_id,
)
from lobster.core.utils.h5ad_utils import validate_for_h5ad
from lobster.core.utils.layer_keys import layer_names

# Import for IR support (TYPE_CHECKING to avoid circular import)
if TYPE_CHECKING:
    from anndata import AnnData
    from plotly.graph_objects import Figure

    from lobster.core.provenance.analysis_ir import AnalysisStep
else:
    AnnData = Any
    Figure = Any

# Try to import ProteomicsAdapter - may not be available in public distribution
try:
    from lobster.core.adapters.proteomics_adapter import ProteomicsAdapter

    PROTEOMICS_AVAILABLE = True
except ImportError:
    PROTEOMICS_AVAILABLE = False
    ProteomicsAdapter = None

# Import MuData backend if available
try:
    from lobster.core.backends.mudata_backend import MuDataBackend

    MUDATA_BACKEND_AVAILABLE = True
except ImportError:
    MUDATA_BACKEND_AVAILABLE = False

# Import MuData if available
try:
    import mudata

    MUDATA_AVAILABLE = True
except ImportError:
    MUDATA_AVAILABLE = False
    mudata = None

logger = logging.getLogger(__name__)

_anndata_module = None
_nbformat_module = None
_plotly_go = None
_plotly_io = None


def _ensure_anndata():
    """Lazily import anndata with a helpful error message."""

    global _anndata_module
    if _anndata_module is None:
        try:
            import anndata as _anndata_module  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            from lobster.core.component_registry import get_install_command

            cmd = get_install_command("anndata")
            raise ImportError(
                f"anndata is required for DataManagerV2 operations. "
                f"Install with: {cmd}"
            ) from exc
    return _anndata_module


def _is_anndata_instance(obj: Any) -> bool:
    try:
        module = _ensure_anndata()
    except ImportError:
        return False
    return isinstance(obj, module.AnnData)


def _ensure_nbformat():
    """Lazily import nbformat only when needed."""

    global _nbformat_module
    if _nbformat_module is None:
        try:
            import nbformat as _nbformat_module  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            from lobster.core.component_registry import get_install_command

            cmd = get_install_command("nbformat")
            raise ImportError(
                f"nbformat is required for notebook export features. "
                f"Install with: {cmd}"
            ) from exc
    return _nbformat_module


def _ensure_plotly():
    """Lazily import Plotly graph objects and IO helpers."""

    global _plotly_go, _plotly_io
    if _plotly_go is None or _plotly_io is None:
        try:
            import plotly.graph_objects as go_module  # type: ignore
            import plotly.io as pio_module  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            from lobster.core.component_registry import get_install_command

            cmd = get_install_command("plotly")
            raise ImportError(
                f"Plotly is required for visualization features. "
                f"Install with: {cmd}"
            ) from exc
        _plotly_go = go_module
        _plotly_io = pio_module
    return _plotly_go, _plotly_io


BACKED_MODE_THRESHOLD_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB


def _close_backed_handle(adata: Any) -> None:
    """Best-effort close of HDF5 file handle on backed AnnData."""
    if getattr(adata, "isbacked", False) and hasattr(adata, "file"):
        try:
            adata.file.close()
        except Exception:  # nosec B110 # Best-effort backed-file close during cleanup.
            pass


#: Default in-memory budget for the modality registry, in GiB.
#:
#: The registry uses a byte budget because a count cap cannot bound the memory used by
#: differently sized modalities. Spilling least-recently-used entries to disk keeps the
#: resident cache bounded while allowing evicted modalities to be reloaded.
DEFAULT_MODALITY_MEMORY_BUDGET_GB = 8.0


def _resolve_modality_memory_budget() -> int:
    """Resolve the modality registry memory budget in bytes (0 disables it)."""
    raw = os.environ.get("LOBSTER_MODALITY_MEMORY_BUDGET_GB")
    if raw is None:
        budget_gb = DEFAULT_MODALITY_MEMORY_BUDGET_GB
    else:
        try:
            budget_gb = float(raw)
        except ValueError:
            logger.warning(
                f"Invalid LOBSTER_MODALITY_MEMORY_BUDGET_GB={raw!r}; "
                f"falling back to {DEFAULT_MODALITY_MEMORY_BUDGET_GB} GiB"
            )
            budget_gb = DEFAULT_MODALITY_MEMORY_BUDGET_GB
    if budget_gb <= 0:
        return 0
    return int(budget_gb * 1024**3)


def _matrix_nbytes(matrix: Any) -> int:
    """Best-effort byte size of a dense or sparse matrix.

    Mock-safe: unit tests store ``Mock`` AnnData objects whose ``nbytes`` is a
    ``Mock``, which must contribute 0 rather than raising.
    """
    if matrix is None:
        return 0

    def _attr_bytes(obj: Any, attr: str) -> int:
        value = getattr(obj, attr, None)
        nbytes = getattr(value, "nbytes", None)
        if nbytes is None or isinstance(nbytes, Mock):
            return 0
        try:
            return int(nbytes)
        except (TypeError, ValueError):
            return 0

    try:
        # Sparse: data + indices + indptr (CSR/CSC) or data + row/col (COO)
        if hasattr(matrix, "nnz"):
            total = sum(
                _attr_bytes(matrix, attr)
                for attr in ("data", "indices", "indptr", "row", "col")
            )
            if total:
                return total
            nnz = getattr(matrix, "nnz", 0)
            if isinstance(nnz, Mock):
                return 0
            return int(nnz) * 12  # ~4B index + 8B value

        nbytes = getattr(matrix, "nbytes", None)
        if nbytes is not None and not isinstance(nbytes, Mock):
            return int(nbytes)

        size = getattr(matrix, "size", None)
        dtype = getattr(matrix, "dtype", None)
        if (
            size is not None
            and dtype is not None
            and not isinstance(size, Mock)
            and hasattr(dtype, "itemsize")
        ):
            return int(size) * int(dtype.itemsize)
    except Exception:  # pragma: no cover - defensive, never fail a store
        return 0
    return 0


def estimate_anndata_bytes(adata: Any) -> int:
    """Estimate the resident memory held by an AnnData object.

    Counts ``.X``, ``.layers``, ``.obsm``/``.varm``, ``.obsp``/``.varp`` and
    ``.raw`` — the containers that hold matrix-scale data.  ``.obs``/``.var``
    dataframes are small by comparison and are deliberately not walked, so this
    is a lower bound; it is used only to decide *when* to evict.

    Backed AnnData contributes 0 for ``.X`` because that matrix lives on disk,
    which is the correct answer for an in-memory budget.
    """
    if adata is None:
        return 0
    if getattr(adata, "isbacked", False):
        # .X is an on-disk HDF5 dataset; only in-memory annotations count.
        total = 0
    else:
        total = _matrix_nbytes(getattr(adata, "X", None))

    for container in ("layers", "obsm", "varm", "obsp", "varp"):
        mapping = getattr(adata, container, None)
        if mapping is None or isinstance(mapping, Mock):
            continue
        try:
            items = list(mapping.items())
        except (
            Exception
        ):  # nosec B112 # Ignore unreadable optional metadata mappings during inspection.
            continue
        for key, value in items:
            # anndata >=0.13 exposes .X inside .layers under the key None.
            # Counting it here would double every matrix in the registry and
            # make the budget silently twice as tight as configured.
            if container == "layers" and key is None:
                continue
            total += _matrix_nbytes(value)

    raw = getattr(adata, "raw", None)
    if raw is not None and not isinstance(raw, Mock):
        total += _matrix_nbytes(getattr(raw, "X", None))

    return total


def _png_readme_line(include_png: bool, png_included: int, png_not_copied: int) -> str:
    """Inventory-driven PNG wording: NOT generated locally is not the same as NOT preserved.

    ``include_png=False`` (``--no-png``) only disables generation; recorded PNG renditions
    are always retained.
    """
    generation = (
        "no new PNG was rendered locally (--no-png: generation disabled)"
        if not include_png
        else "no new PNG was rendered locally"
    )
    if png_included:
        recorded = (
            f"{png_included} already recorded PNG rendition(s) included with their exact "
            "bytes (`png_renditions` in `plots/index.json`)"
        )
    elif png_not_copied:
        recorded = "no recorded PNG rendition could be copied"
    else:
        recorded = "no PNG rendition is recorded for these revisions"
    line = f"{recorded}; {generation}"
    if png_not_copied:
        line += (
            f". {png_not_copied} recorded PNG rendition(s) were NOT copied: see "
            "'Not copied' below"
        )
    if not png_included and not png_not_copied:
        line += (
            " (each index entry records `png: not_rendered_locally`). Open the HTML "
            "view to export a PNG"
        )
    return line


def _not_copied_section(
    plots_index: List[Dict[str, Any]], legacy_names: Optional[List[str]] = ()
) -> str:
    """Visible list of everything this package could not copy (never only in index.json).

    ``legacy_names``: flat (legacy) files in ``plots/``, which this export does not
    support; ``None`` = they could not be listed.
    """
    from lobster.core.figures.listing import safe_label

    lines = []
    for info in plots_index:
        ref = info.get("figure_ref")
        if info.get("error"):
            lines.append(f"- {ref}: figure not exported ({info['error']})")
        if info.get("html_error"):
            lines.append(f"- {ref}: HTML view not written ({info['html_error']})")
        for png in info.get("png_errors", []):
            lines.append(
                f"- {ref}: recorded PNG {png['rendition_id']} not copied ({png['reason']})"
            )
    if legacy_names is None:
        lines.append("- plots/: legacy plot files could not be listed")
    for name in legacy_names or ():
        lines.append(
            f"- plots/{safe_label(name) or 'unnamed'}: legacy plot file, not supported "
            "by this export (unchanged in the workspace)"
        )
    if not lines:
        return ""
    return (
        "\n## Not copied\n\nThese saved items are NOT in this package; their state is "
        "unchanged in the workspace.\n\n" + "\n".join(lines) + "\n"
    )


_SAFE_REF_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


def _ref_id(value: Any) -> str:
    """A stored reference ID as display-safe text; anything else is ``unnamed``."""
    if isinstance(value, str) and _SAFE_REF_ID.fullmatch(value):
        return value
    return "unnamed"


def _lifecycle_refs_not_included(store: Any, record: Dict[str, Any]) -> List[Dict]:
    """The lifecycle artifacts a revision references that a view package does NOT copy.

    Read from the revision record and its execution record only: nothing is guessed. When
    the execution record cannot be read, its entry carries ``unread`` (a safe reason) and
    no code/activity entry is invented.
    """
    from lobster.core.figures.errors import safe_reason

    refs: List[Dict[str, str]] = []
    source = record.get("source")
    if isinstance(source, dict) and source.get("source_id"):
        refs.append({"kind": "source", "id": _ref_id(source["source_id"])})
    execution = record.get("execution")
    execution_id = (
        execution.get("execution_id") if isinstance(execution, dict) else None
    )
    if not execution_id:
        return refs
    try:
        executed = store.read_execution(execution_id)
    except Exception as e:
        refs.append(
            {"kind": "execution", "id": _ref_id(execution_id), "unread": safe_reason(e)}
        )
        return refs
    refs.append({"kind": "execution", "id": _ref_id(execution_id)})
    if executed.get("code_id"):
        refs.append({"kind": "code", "id": _ref_id(executed["code_id"])})
    if executed.get("activity_record"):
        refs.append({"kind": "activity", "id": _ref_id(executed.get("activity_id"))})
    return refs


def _describe_refs(refs: List[Dict[str, str]]) -> str:
    """``source <id>; execution <id>; ...`` (an unread execution says what is unknown)."""
    parts = []
    for item in refs:
        part = f"{item['kind']} {item['id']}"
        if item.get("unread"):
            part += f" (code/activity references unknown ({item['unread']}))"
        parts.append(part)
    return "; ".join(parts)


_VIEW_PACKAGE_SCOPE = (
    "This is a view package: each saved figure revision is copied as its exact "
    "`revision.json`, `figure.json` and `config.json` bytes, with its views. The "
    "lifecycle artifacts those revisions reference (source data, execution records, "
    "code, activity records) are NOT copied, so this package is not a complete, "
    "reloadable figure store. `provenance.json`, when present, is a session summary, "
    "not those artifacts. They stay unchanged in the workspace."
)


def _not_included_section(plots_index: List[Dict[str, Any]]) -> str:
    """Package scope + the per-revision lifecycle references this package does not copy."""
    if not plots_index:
        return ""
    lines = [
        f"- {info.get('figure_ref')}: {_describe_refs(info['not_included'])}"
        for info in plots_index
        if info.get("not_included")
    ]
    scope = "\n## Package scope\n\n" + _VIEW_PACKAGE_SCOPE + "\n"
    if not lines:
        return scope
    return (
        scope
        + "\n## Not included: referenced lifecycle artifacts\n\nReferenced by the "
        "revisions in this package and NOT copied into it:\n\n"
        + "\n".join(lines)
        + "\n"
    )


#: Prefix of the auto-save result entry that counts modalities NOT saved because their
#: target is inside the managed figure folder (callers must not report them as saved).
AUTOSAVE_NOT_SAVED_PREFIX = "Not saved: "
_AUTOSAVE_REFUSED_LOG = (
    "Auto-save skipped: the target is inside the managed figure folder."
)
_SPILL_REFUSED_LOG = (
    "Memory spill skipped: the spill file is inside the managed figure folder."
)
# A name only classified (never created) to ask "may a file be written in here?".
_DIR_CHECK_CHILD = ".export-dir-check"


def _managed_refusal() -> ManagedPathError:
    from lobster.core.figures.export_guard import MANAGED_EXPORT_REASON

    return ManagedPathError(MANAGED_EXPORT_REASON)


def _assert_container_allowed(workspace: Path, directory: Path) -> None:
    """Refuse unless files may be written in ``directory`` (an absolute spelling)."""
    from lobster.core.figures.export_guard import assert_export_allowed

    probe = directory / _DIR_CHECK_CHILD
    try:
        os.lstat(probe)
    except (FileNotFoundError, NotADirectoryError):
        pass  # absent: its classification resolves the folder itself
    except Exception:  # cannot tell whether it exists: refuse (fail closed)
        raise _managed_refusal() from None
    else:  # an existing entry there could point the check elsewhere: refuse
        raise _managed_refusal() from None
    assert_export_allowed(workspace, probe)


def _assert_dirs_allowed(workspace: Path, directory: Union[str, Path]) -> None:
    """Raise ManagedPathError unless ``directory`` may be created recursively.

    The final folder, existing or not, is checked as the place files are written in;
    every lexical prefix of its UNnormalized absolute spelling (``..`` not collapsed)
    that does not exist yet is checked both as a new entry and as such a folder. A
    relative spelling is taken against the current directory, as mkdir takes it.
    """
    from lobster.core.figures.export_guard import assert_export_allowed

    spelled = Path(directory)
    absolute = spelled if spelled.is_absolute() else Path.cwd() / spelled
    _assert_container_allowed(workspace, absolute)
    parts = absolute.parts
    for end in range(1, len(parts) + 1):
        prefix = Path(*parts[:end])
        if not os.path.exists(prefix):
            assert_export_allowed(workspace, prefix)
            _assert_container_allowed(workspace, prefix)


def _package_staging_parent() -> str:
    """The folder a data package is built in, chosen WITHOUT writing anywhere.

    ``tempfile.gettempdir()`` creates and deletes a probe file in each candidate until
    one is writable, so it is not used before the choice is checked. The choice is the
    one already made by ``tempfile`` if any, else the first of its candidates (TMPDIR,
    TEMP, TMP, the platform folders, the current directory) that is a writable folder.
    """
    if tempfile.tempdir is not None:
        return tempfile.tempdir
    candidates = [os.environ.get(n) for n in ("TMPDIR", "TEMP", "TMP")]
    if os.name == "nt":
        candidates += [
            os.path.expanduser(r"~\AppData\Local\Temp"),
            os.path.expandvars(r"%SYSTEMROOT%\Temp"),
            r"c:\temp",
            r"c:\tmp",
            r"\temp",
            r"\tmp",
        ]
    else:
        # tempfile's own platform candidates. They only choose the parent: the
        # package is staged in a new folder that tempfile creates in it (mkdtemp).
        candidates += ["/tmp", "/var/tmp", "/usr/tmp"]  # nosec B108
    candidates.append(os.getcwd())
    for candidate in candidates:
        if (
            candidate
            and os.path.isdir(candidate)
            and os.access(candidate, os.W_OK | os.X_OK)
        ):
            return os.path.abspath(candidate)
    return os.getcwd()


def _assert_not_a_folder(path: Path) -> None:
    """Refuse a copy destination that is an existing folder: ``shutil.copy2`` would
    write an unchecked child name inside it."""
    if os.path.isdir(path):
        raise _managed_refusal() from None


def _autosave_file_name(modality_name: str) -> str:
    """The auto-save file of a modality, relative to data_dir (never double-suffixed)."""
    if modality_name.endswith("_autosave"):
        return f"{modality_name}.h5ad"
    return f"{modality_name}_autosave.h5ad"


def _is_mudata_backend(backend: Any) -> bool:
    """True for the native MuData backend, the one writer that makes a backup."""
    return (
        MUDATA_BACKEND_AVAILABLE
        and isinstance(MuDataBackend, type)  # replaced in some tests
        and isinstance(backend, MuDataBackend)
    )


def _package_payload_names(name: str) -> tuple:
    return (f"{name}.h5ad", f"{name}_metadata.json")


def _is_plain_file_name(name: str) -> bool:
    separators = {"/", os.sep} | ({os.altsep} if os.altsep else set())
    return name not in {"", ".", ".."} and not any(c in name for c in separators)


def _safe_filename(name: str) -> str:
    """Reduce a modality name to a filesystem-safe stem."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)
    return safe.strip("._") or "modality"


def _should_use_backed_mode(path: Path) -> bool:
    """Return True if h5ad file is large enough to warrant backed loading."""
    try:
        return path.stat().st_size >= BACKED_MODE_THRESHOLD_BYTES
    except OSError:
        return False


def _is_backed_mode_enabled() -> bool:
    """Check if backed-mode loading is enabled via env var."""
    return os.environ.get("LOBSTER_BACKED_MODE", "0") == "1"


def _read_h5ad_smart(
    path: Path,
    force_backed: bool = False,
    backed_mode_enabled: bool = False,
) -> "AnnData":
    """Load h5ad with automatic backed-mode for large files.

    Files >= 1 GB load with backed='r' to avoid OOM.  The returned AnnData
    has .X as an HDF5 dataset — callers that mutate .X must call
    ``DataManagerV2.materialize_modality()`` first.

    Backed mode must be explicitly enabled via ``backed_mode_enabled`` param
    or ``LOBSTER_BACKED_MODE=1`` env var.
    """
    anndata_mod = _ensure_anndata()
    enabled = backed_mode_enabled or _is_backed_mode_enabled()
    use_backed = force_backed or (enabled and _should_use_backed_mode(path))
    if use_backed:
        try:
            size_gb = path.stat().st_size / (1024**3)
            logger.info(f"Loading '{path.name}' in backed mode ({size_gb:.1f} GB)")
        except OSError:
            logger.info(f"Loading '{path.name}' in backed mode")
        adata = anndata_mod.read_h5ad(path, backed="r")
    else:
        adata = anndata_mod.read_h5ad(path)
    from lobster.core.utils.h5ad_utils import repair_log1p_base

    repair_log1p_base(adata)
    return adata


class MetadataEntry(TypedDict, total=False):
    """
    TypedDict for GEO metadata store entries.

    Enforces consistent structure across all metadata storage locations
    to prevent KeyError bugs when retrieving metadata.

    Structure:
        metadata (Dict[str, Any]): The actual GEO metadata from GEOparse
        validation_result (Dict[str, Any]): Validation information (optional)
        fetch_timestamp (str): ISO format timestamp when metadata was fetched
        strategy_config (Dict[str, Any]): Download strategy configuration (optional)
        stored_by (str): Component that stored the metadata
        modality_detection (Dict[str, Any]): Modality detection results (optional)
        concatenation_decision (Dict[str, Any]): Concatenation strategy (optional)
    """

    metadata: Dict[str, Any]
    validation_result: Dict[str, Any]
    fetch_timestamp: str
    strategy_config: Dict[str, Any]
    stored_by: str
    modality_detection: Dict[str, Any]
    concatenation_decision: Dict[str, Any]


class SuppressKaleidoLogging:
    """Context manager to temporarily suppress verbose Kaleido logging during image generation."""

    def __init__(self):
        self.kaleido_logger = logging.getLogger("kaleido")
        self.original_level = None

    def __enter__(self):
        self.original_level = self.kaleido_logger.level
        self.kaleido_logger.setLevel(logging.WARNING)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_level is not None:
            self.kaleido_logger.setLevel(self.original_level)


class DataManagerV2:
    """
    Modular data manager for multi-omics analysis.

    This class orchestrates modality adapters, storage backends,
    and validation to provide a unified interface for managing
    multi-modal biological data with complete provenance tracking.
    """

    def __init__(
        self,
        default_backend: str = "h5ad",
        workspace_path: Optional[Union[str, Path]] = None,
        enable_provenance: bool = True,
        session_dir: Optional[Union[str, Path]] = None,
        console=None,
        auto_scan: bool = True,
        backed_mode: bool = False,
    ):
        """
        Initialize DataManagerV2.

        Args:
            default_backend: Default storage backend to use
            workspace_path: Optional workspace directory for data storage.
                Resolution order: explicit path > LOBSTER_WORKSPACE env var > cwd/.lobster_workspace
            enable_provenance: Whether to enable provenance tracking
            session_dir: Optional path to session directory for provenance persistence.
                When provided, provenance is persisted to disk and survives process restarts.
            console: Optional Rich console instance for progress tracking
            auto_scan: Whether to automatically scan workspace for available datasets
            backed_mode: Enable backed-mode loading for large h5ad files (>1GB).
                Also enabled by LOBSTER_BACKED_MODE=1 env var.
        """
        self.default_backend = default_backend
        self.workspace_path = resolve_workspace(
            explicit_path=workspace_path, create=True
        )
        self.enable_provenance = enable_provenance
        self._session_dir = Path(session_dir) if session_dir else None
        self.console = console  # Store console for progress tracking in tools
        self.backed_mode = backed_mode or _is_backed_mode_enabled()

        # Core storage
        self.backends: Dict[str, IDataBackend] = {}
        self.adapters: Dict[str, IModalityAdapter] = {}
        # OrderedDict, not dict: LRU ordering needs move_to_end().  The guards
        # around move_to_end() elsewhere assumed a plain dict here, so recency
        # was never actually tracked before the modality backend was attached.
        self.modalities: Dict[str, "AnnData"] = OrderedDict()

        # WorkspaceModalityBackend integration (feature-flagged)
        self._modality_backend: Optional[Any] = None
        self._modality_cache_cap: int = 8

        # Bound the modality registry by resident bytes and spill the
        # least-recently-used entries to h5ad.  Eviction already existed, but
        # only ran when a WorkspaceModalityBackend was attached, and nothing in
        # the shipped code path ever attaches one, so it never ran in
        # production.  Spilled modalities stay *retrievable*: get_modality()
        # reloads them on demand, so capping memory cannot turn an OOM into a
        # "modality not found".
        self._modality_memory_budget: int = _resolve_modality_memory_budget()
        self._modality_spilled: Dict[str, Path] = {}
        self._modality_spill_dir = self.workspace_path / ".lobster" / "spill"

        # Track modality sources and dirty state for efficient auto-save
        self._modality_sources: Dict[str, Path] = (
            {}
        )  # modality_name -> source file path
        self._modality_dirty: set = set()  # modalities modified since load

        # Workspace restoration attributes
        self.session_file = self.workspace_path / ".session.json"
        self._available_datasets: Dict[str, Dict] = {}
        self._auto_scan_on_access = auto_scan
        self.session_data: Optional[Dict] = None
        self.session_id = str(datetime.now().timestamp())  # Unique session ID

        # Cache workspace scan results with 30-second TTL (Time To Live)
        self._available_datasets_cache: Optional[Dict[str, Dict]] = None
        self._scan_timestamp: float = 0  # Timestamp of last scan
        self._scan_ttl: int = 30  # Cache TTL in seconds (configurable)
        self._timing_enabled = False
        self._latest_timings: Dict[str, float] = {}
        self.profile_timings_enabled = False

        # Metadata storage for GEO datasets and other sources temporary until data is loaded
        self.metadata_store: Dict[str, Dict[str, Any]] = {}

        # Download queue for dataset downloads (research_agent → data_expert handoff)
        # Lazily initialized queue directories
        self._queues_dir = self.workspace_path / ".lobster" / "queues"
        self._download_queue = None
        self._publication_queue = None
        self._publication_queue_unavailable = False

        # Processing log for user-facing messages
        self.processing_log: List[str] = []

        # Plot management (delegated to PlotManager via lazy property)
        self._plot_manager: Optional[PlotManager] = None
        self.max_plots_history: int = 50  # Passed to PlotManager on initialization

        # Multi-process safe session file access
        self._session_lock = threading.Lock()
        self._session_lock_path = self.session_file.with_suffix(".lock")

        # Thread-safety for mutable state accessed by concurrent tool calls
        # (e.g. execute_custom_code via asyncio.to_thread)
        self._state_lock = threading.Lock()

        # Client session identity shown in notebook exports. Only the owning client
        # binds it (bind_client_session); the timestamp session_id above is not it.
        self._client_session_lock = threading.Lock()
        self._client_session_id: Optional[str] = None
        self._client_session_unknown = False

        # Optional callback for modality loaded events: fn(name, adata)
        self.on_modality_loaded = None

        # Locality provider for materializing modalities to local disk
        # (enables subprocess-based execution on cloud where data lives in S3)
        self._locality_provider = None

        # Current dataset management (for legacy compatibility)
        self.current_dataset: Optional[str] = None  # Name of current active modality
        self.current_data: Optional[pd.DataFrame] = None  # Legacy compatibility
        self.current_metadata: Dict[str, Any] = {}  # Legacy metadata
        self.adata: Optional["AnnData"] = None  # Legacy AnnData reference

        # Provenance tracking
        self.provenance = (
            ProvenanceTracker(session_dir=self._session_dir)
            if enable_provenance
            else None
        )

        # Notebook-based pipeline support (lazy initialization to avoid circular imports)
        self._notebook_exporter = None
        self._notebook_executor = None
        self._enable_notebooks = enable_provenance

        # Setup workspace
        self._setup_workspace()

        # Scan workspace for available datasets
        # Workspace scanning is deferred until first access via get_available_datasets()

        # Load session metadata if exists
        if self.session_file.exists():
            self._load_session_metadata()

        # Register default backends and adapters
        self._register_default_backends()
        self._register_default_adapters()

        # Auto-load existing modalities from workspace (session persistence)
        if auto_scan:
            self._auto_load_modalities()

        logger.debug(f"Initialized DataManagerV2 with workspace: {self.workspace_path}")

    def _get_queues_dir(self) -> Path:
        """Ensure the queues directory exists before use."""

        _assert_dirs_allowed(self.workspace_path, self._queues_dir)
        self._queues_dir.mkdir(parents=True, exist_ok=True)
        return self._queues_dir

    @property
    def download_queue(self):  # type: ignore[override]
        if self._download_queue is None:
            from lobster.core.download_queue import DownloadQueue

            queue_file = self._get_queues_dir() / "download_queue.jsonl"
            self._download_queue = DownloadQueue(queue_file=queue_file)
        return self._download_queue

    @property
    def publication_queue(self):  # type: ignore[override]
        if self._publication_queue_unavailable:
            return None
        if self._publication_queue is None:
            from lobster.core.component_registry import component_registry

            PublicationQueue = component_registry.get_service("publication_queue")
            if PublicationQueue is None:
                logger.debug(
                    "Publication queue feature not available (premium feature)"
                )
                self._publication_queue_unavailable = True
                return None

            queue_file = self._get_queues_dir() / "publication_queue.jsonl"
            self._publication_queue = PublicationQueue(queue_file=queue_file)
        return self._publication_queue

    @property
    def locality_provider(self):
        """Lazy-initialized data locality provider for materializing modalities."""
        if self._locality_provider is None:
            from lobster.core.interfaces.locality import LocalDataLocalityProvider

            self._locality_provider = LocalDataLocalityProvider(self)
        return self._locality_provider

    def set_locality_provider(self, provider) -> None:
        """Register a custom locality provider (e.g., S3-backed for cloud)."""
        self._locality_provider = provider

    @property
    def figure_store(self):
        """The managed-figure authority for this workspace (contract §9).

        Local default: :class:`lobster.core.figures.store.LocalFigureStore`. Cloud injects a
        catalog-backed store with :meth:`set_figure_store`.
        """
        store = getattr(self, "_figure_store", None)
        if store is None:
            from lobster.core.figures.store import LocalFigureStore

            store = LocalFigureStore(
                self.workspace_path, session_id=str(self.session_id)
            )
            self._figure_store = store
        return store

    def set_figure_store(self, store) -> None:
        """Register a custom figure store (e.g., Cloud catalog-backed)."""
        self._figure_store = store

    def resolve_input_identity(self, modality_name: str):
        """Return the verified :class:`InputEdge` for a loaded modality, or None (contract §5.2).

        Resolution order (as implemented): provider object identity → lineage
        ``instance_id`` (store event) → None. Never random, never by name alone.
        """
        if modality_name not in self.modalities:
            return None
        return self.input_identity_of(modality_name, self.modalities[modality_name])

    def input_identity_of(self, modality_name: str, adata: Any = None):
        """The :class:`InputEdge` of ``adata``, the object actually used for ``modality_name``.

        Same resolution order as :meth:`resolve_input_identity` (provider object identity →
        the object's lineage ``instance_id``, a store event → None), applied to the given
        object instead of a later lookup by name. ``adata=None`` (a provider file the data
        manager does not hold) can only have a provider identity. Never random, never by
        name alone; identifies a store event or stored object, not live content.
        """
        from lobster.core.figures.records import InputEdge

        provider = self._locality_provider
        if provider is not None and hasattr(provider, "dataset_identity"):
            try:
                ident = provider.dataset_identity(modality_name)
            except Exception:  # provider failure must not invent identity
                ident = None
            if ident and ident.get("instance_id"):
                return InputEdge(
                    modality_name,
                    str(ident["instance_id"]),
                    str(ident.get("origin", "provider_object_version")),
                    True,
                )
        if adata is None:
            return None
        lineage = (
            getattr(adata, "uns", {}).get("lineage") if hasattr(adata, "uns") else None
        )
        instance_id = lineage.get("instance_id") if isinstance(lineage, dict) else None
        if instance_id:
            return InputEdge(modality_name, str(instance_id), "store_event", True)
        return None

    @property
    def client_session_id(self) -> Optional[str]:
        """Client session id bound by the owning client, or None when unknown."""
        with self._client_session_lock:
            return self._client_session_id

    def bind_client_session(
        self, session_id: Any, *, replace: bool = False
    ) -> Optional[str]:
        """Bind the owning client's session id for notebook export; never raises.

        The first valid id binds. The same id again is a no-op. A different id,
        or an id outside the display allowlist, makes the identity unknown, and
        it stays unknown until a valid ``replace=True`` bind (session restore).
        Returns the resulting identity (None when unknown). Any failure while
        handling the value also leaves the identity unknown, never the old one.
        """
        reason = None
        result = None
        # A plain str copy of the checked text, or None; never raises.
        identity = _display_client_session_id(session_id)
        if identity is None:
            reason = "invalid_type"
            try:
                if isinstance(session_id, str):
                    reason = "invalid_format"
            except Exception:
                pass
        try:
            with self._client_session_lock:
                try:
                    if identity is None:
                        self._client_session_id = None
                        self._client_session_unknown = True
                    elif replace:
                        self._client_session_id = identity
                        self._client_session_unknown = False
                    elif self._client_session_unknown:
                        pass
                    elif self._client_session_id is None:
                        self._client_session_id = identity
                    elif self._client_session_id != identity:
                        reason = "conflict"
                        self._client_session_id = None
                        self._client_session_unknown = True
                except Exception:
                    reason = "internal"
                    self._client_session_id = None
                    self._client_session_unknown = True
                result = self._client_session_id
        except Exception:
            # The identity lock itself failed; leave no old identity behind.
            reason, result = "internal", None
            try:
                self._client_session_unknown = True
                self._client_session_id = None
            except Exception:
                pass
        if reason:
            logger.warning(
                "Notebook export: client session identity is unknown (%s)", reason
            )
        return result

    @property
    def notebook_exporter(self):
        """Lazy initialize notebook exporter."""
        if self._notebook_exporter is None and self._enable_notebooks:
            from lobster.core.notebook_exporter import NotebookExporter

            self._notebook_exporter = NotebookExporter(self.provenance, self)
        return self._notebook_exporter

    @property
    def notebook_executor(self):
        """Lazy initialize notebook executor."""
        if self._notebook_executor is None:
            from lobster.core.notebook_executor import NotebookExecutor

            self._notebook_executor = NotebookExecutor(self)
        return self._notebook_executor

    @property
    def plot_manager(self) -> PlotManager:
        """Lazy initialize PlotManager for visualization management."""
        if self._plot_manager is None:
            self._plot_manager = PlotManager(
                workspace_path=self.workspace_path,
                max_plots_history=self.max_plots_history,
                enable_ir=self.provenance is not None,
            )
            self._plot_manager.configure_figure_lifecycle(
                store_provider=lambda: self.figure_store,
                identity_resolver=self.resolve_input_identity,
                session_id_provider=lambda: self.session_id,
            )
        return self._plot_manager

    @property
    def latest_plots(self) -> List[Dict[str, Any]]:
        """Access latest_plots via PlotManager for backward compatibility."""
        return self.plot_manager.latest_plots

    @property
    def visualization_state(self) -> Dict[str, Any]:
        """Access visualization_state via PlotManager for backward compatibility."""
        return self.plot_manager.visualization_state

    def _setup_workspace(self) -> None:
        """Set up workspace directories."""
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.data_dir = self.workspace_path / "data"
        self.exports_dir = self.workspace_path / "exports"
        self.cache_dir = self.workspace_path / "cache"
        self.literature_cache_dir = self.workspace_path / "literature_cache"

        for directory in [
            self.data_dir,
            self.exports_dir,
            self.cache_dir,
            self.literature_cache_dir,
        ]:
            directory.mkdir(exist_ok=True)

        # Create literature cache subdirectories
        (self.literature_cache_dir / "publications").mkdir(exist_ok=True)
        (self.literature_cache_dir / "parsed_docs").mkdir(exist_ok=True)

    def _register_default_backends(self) -> None:
        """Register default storage backends."""
        # H5AD backend
        h5ad_backend = H5ADBackend(base_path=self.data_dir)
        h5ad_backend.guard_workspace = self.workspace_path
        self.register_backend("h5ad", h5ad_backend)

        # MuData backend if available
        if MUDATA_BACKEND_AVAILABLE:
            mudata_backend = MuDataBackend(base_path=self.data_dir)
            # This data manager's own instance never backs up into plots/<figure>/
            mudata_backend.guard_workspace = self.workspace_path
            self.register_backend("mudata", mudata_backend)

    def _register_default_adapters(self) -> None:
        """Register default modality adapters.

        Construction is shared with exported notebooks (see
        ``lobster.core.adapters.factory``): entry-point adapters first, then the
        built-in fallbacks, so a notebook builds the same adapter as the session.
        """
        for name, adapter in create_default_adapters().items():
            if name not in self.adapters:
                self.register_adapter(name, adapter)

    def _resolve_adapter_for_file(self, file_path: Path):
        """Resolve the best adapter for a file based on extension.

        Returns (adapter_name, adapter_instance) or None if no adapter matches.

        Strategy:
        - h5ad: return None (handled by direct anndata.read_h5ad, faster)
        - Unambiguous format (only one adapter supports it): use that adapter
        - Ambiguous format (multiple adapters): use a BaseAdapter for generic loading
        """
        ext = file_path.suffix.lower().lstrip(".")
        # Compound extensions
        if file_path.name.endswith(".vcf.gz"):
            ext = "vcf.gz"

        if ext == "h5ad":
            return None  # Caller handles h5ad directly

        # Find all adapters that support this extension
        matching = []
        for name, adapter in self.adapters.items():
            if ext in adapter.get_supported_formats():
                matching.append((name, adapter))

        if len(matching) == 1:
            return matching[0]
        elif len(matching) > 1:
            # Ambiguous — use BaseAdapter for generic loading
            return ("generic", BaseAdapter())

        # No registered adapter, but BaseAdapter might handle it
        generic = BaseAdapter()
        if ext in generic.get_supported_formats():
            return ("generic", generic)

        return None

    def _auto_load_modalities(self) -> None:
        """
        Auto-load existing modalities from workspace data directory.

        Loads H5AD files (via anndata.read_h5ad) and tabular files
        (CSV, TSV, parquet, XLSX) via the adapter system.

        Behavior:
        - Scans workspace/data/ for all supported file formats
        - H5AD files loaded first (they take precedence over same-stem tabular files)
        - Tabular files: 500MB size guard (expand significantly in memory)
        - Skips H5AD files >2GB (memory safety)
        - Silently skips corrupted/incompatible files (logs warning)
        """
        if not self.data_dir.exists():
            return

        # --- Phase 1: Collect all loadable files ---
        h5ad_files = list(self.data_dir.glob("*.h5ad"))

        scannable = self._get_scannable_extensions()
        tabular_to_scan = self._TABULAR_FORMATS & scannable

        tabular_files = []
        for ext in sorted(tabular_to_scan):
            tabular_files.extend(self.data_dir.glob(f"*.{ext}"))

        if not h5ad_files and not tabular_files:
            return

        total_files = len(h5ad_files) + len(tabular_files)
        logger.info(
            f"Auto-loading from workspace: {len(h5ad_files)} h5ad, "
            f"{len(tabular_files)} tabular ({total_files} total)..."
        )

        loaded_count = 0
        skipped_large = 0
        skipped_failed = 0
        corrupted_autosaves_cleaned = 0
        corrupted_regular_files = []

        # --- Phase 2: H5AD files first (existing logic, byte-for-byte) ---
        for h5ad_file in h5ad_files:
            try:
                file_size_gb = h5ad_file.stat().st_size / (1024**3)
                if file_size_gb > 4.0:
                    skipped_large += 1
                    logger.debug(
                        f"Skipped very large file {h5ad_file.name} ({file_size_gb:.1f} GB > 4GB threshold). "
                        f"Use load_modality tool to explicitly load if needed."
                    )
                    continue

                modality_name = h5ad_file.stem

                if modality_name in self.modalities:
                    logger.debug(f"Modality '{modality_name}' already loaded, skipping")
                    continue

                adata = _read_h5ad_smart(
                    h5ad_file, backed_mode_enabled=self.backed_mode
                )

                from lobster.core.lineage import ensure_lineage, has_lineage

                if not has_lineage(adata):
                    adata = ensure_lineage(adata, modality_name)
                    logger.debug(
                        f"Created synthetic lineage for legacy file '{modality_name}'"
                    )

                with self._state_lock:
                    self.modalities[modality_name] = adata
                self._modality_sources[modality_name] = h5ad_file

                loaded_count += 1
                logger.debug(
                    f"Auto-loaded '{modality_name}': {adata.n_obs} obs × {adata.n_vars} vars"
                )

            except HdfReadRefusal:
                raise
            except Exception:
                is_autosave = h5ad_file.name.endswith("_autosave.h5ad")
                if is_autosave:
                    try:
                        h5ad_file.unlink()
                        corrupted_autosaves_cleaned += 1
                        logger.debug(f"Removed corrupted autosave: {h5ad_file.name}")
                    except OSError as unlink_err:
                        logger.debug(
                            f"Could not remove corrupted autosave {h5ad_file.name}: {unlink_err}"
                        )
                else:
                    corrupted_regular_files.append(h5ad_file.name)
                continue

        # --- Phase 3: Non-H5AD tabular files ---
        tabular_size_limit_mb = 500

        for data_file in tabular_files:
            try:
                modality_name = data_file.stem

                # H5AD autosave takes precedence — skip if already loaded
                if modality_name in self.modalities:
                    logger.debug(
                        f"Modality '{modality_name}' already loaded (h5ad precedence), skipping {data_file.name}"
                    )
                    continue

                file_size_mb = data_file.stat().st_size / 1e6
                if file_size_mb > tabular_size_limit_mb:
                    skipped_large += 1
                    logger.debug(
                        f"Skipped large tabular file {data_file.name} ({file_size_mb:.0f} MB > {tabular_size_limit_mb}MB threshold)"
                    )
                    continue

                resolved = self._resolve_adapter_for_file(data_file)
                if resolved is None:
                    logger.debug(f"No adapter found for {data_file.name}, skipping")
                    skipped_failed += 1
                    continue

                adapter_name, adapter = resolved
                adata = adapter.from_source(data_file)

                # Guard: skip files that produced no numeric variables (e.g. text-only CSV)
                if adata.n_vars == 0:
                    skipped_failed += 1
                    logger.debug(
                        f"Skipped {data_file.name}: no numeric variables after loading"
                    )
                    continue

                with self._state_lock:
                    self.modalities[modality_name] = adata
                # Track source but do NOT mark dirty — this is the unchanged original
                self._modality_sources[modality_name] = data_file

                loaded_count += 1
                logger.debug(
                    f"Auto-loaded '{modality_name}' via {adapter_name}: "
                    f"{adata.n_obs} obs × {adata.n_vars} vars"
                )

            except HdfReadRefusal:
                raise
            except Exception as e:
                skipped_failed += 1
                logger.warning(f"Could not auto-load {data_file.name}: {e}")
                continue

        # --- Summary logging ---
        if loaded_count > 0:
            logger.info(f"Auto-loaded {loaded_count} modalities from workspace")
        if skipped_large > 0:
            logger.info(
                f"Skipped {skipped_large} large files. Use load_modality tool to load explicitly."
            )
        if corrupted_autosaves_cleaned > 0:
            logger.info(
                f"Cleaned up {corrupted_autosaves_cleaned} corrupted autosave file(s)"
            )
        if corrupted_regular_files:
            logger.warning(
                f"Could not load {len(corrupted_regular_files)} corrupted file(s): "
                f"{', '.join(corrupted_regular_files[:3])}"
                f"{' and more...' if len(corrupted_regular_files) > 3 else ''}"
            )

    @property
    def available_datasets(self) -> Dict[str, Dict]:  # type: ignore[override]
        """Return cached dataset metadata, performing a scan if needed."""

        if (
            self._auto_scan_on_access
            and not self._available_datasets
            and self._available_datasets_cache is None
        ):
            try:
                self.get_available_datasets(force_refresh=True)
            except HdfReadRefusal:
                raise
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug(f"Deferred workspace scan failed: {exc}")
        return self._available_datasets

    @available_datasets.setter
    def available_datasets(self, value: Dict[str, Dict]) -> None:
        self._available_datasets = value

    @contextmanager
    def _measure_step(self, name: str):
        if not self._timing_enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._latest_timings[name] = elapsed

    def enable_timing(self, enabled: bool = True) -> None:
        self._timing_enabled = enabled
        self.profile_timings_enabled = enabled
        if not enabled:
            self._latest_timings = {}

    def get_latest_timings(self, clear: bool = False) -> Dict[str, float]:
        timings = dict(self._latest_timings)
        if clear:
            self._latest_timings = {}
        return timings

    def register_backend(
        self, name: str, backend: IDataBackend, overwrite: bool = False
    ) -> None:
        """
        Register a storage backend.

        Args:
            name: Name for the backend
            backend: Backend implementation
            overwrite: If True, allow overwriting existing backend

        Raises:
            ValueError: If backend name already exists and overwrite=False
        """
        if name in self.backends:
            if overwrite:
                logger.warning(f"Overwriting existing backend: {name}")
            else:
                raise ValueError(
                    f"Backend '{name}' already registered. Use overwrite=True to replace."
                )

        self.backends[name] = backend
        logger.debug(f"Registered backend: {name} ({backend.__class__.__name__})")

    def register_adapter(
        self, name: str, adapter: IModalityAdapter, overwrite: bool = False
    ) -> None:
        """
        Register a modality adapter.

        Args:
            name: Name for the adapter
            adapter: Adapter implementation
            overwrite: If True, allow overwriting existing adapter

        Raises:
            ValueError: If adapter name already exists and overwrite=False
        """
        if name in self.adapters:
            if overwrite:
                logger.warning(f"Overwriting existing adapter: {name}")
            else:
                raise ValueError(
                    f"Adapter '{name}' already registered. Use overwrite=True to replace."
                )

        self.adapters[name] = adapter
        logger.debug(f"Registered adapter: {name} ({adapter.__class__.__name__})")

    def load_modality(
        self,
        name: str,
        source: Union[str, Path, pd.DataFrame, "AnnData"],
        adapter: str,
        validate: bool = True,
        **kwargs,
    ) -> "AnnData":
        """
        Load data for a specific modality.

        Args:
            name: Name for the modality
            source: Data source (file path, DataFrame, or AnnData)
            adapter: Name of adapter to use
            validate: Whether to validate the loaded data
            **kwargs: Additional parameters passed to adapter

        Returns:
            AnnData: Loaded and validated data

        Raises:
            ValueError: If adapter is not registered or validation fails
        """
        # The name becomes part of file names: it must be one plain name.
        check_modality_name(name)
        if adapter not in self.adapters:
            raise ValueError(f"Adapter '{adapter}' not registered")

        adapter_instance = self.adapters[adapter]

        # Load data using adapter
        if not _is_anndata_instance(source):
            adata = adapter_instance.from_source(source, **kwargs)
        else:
            adata = source

        # Validate data if requested
        if validate:
            validation_result = adapter_instance.validate(adata, strict=False)

            if validation_result.has_errors:
                error_msg = f"Validation failed for modality '{name}': {validation_result.errors}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if validation_result.has_warnings:
                logger.warning(
                    f"Validation warnings for modality '{name}': {validation_result.warnings}"
                )

        # Store modality
        with self._state_lock:
            self.modalities[name] = adata
            if hasattr(self.modalities, "move_to_end"):
                self.modalities.move_to_end(name)
            self._modality_dirty.add(name)
            self._enforce_memory_budget()

        # Log provenance
        if self.provenance:
            self.provenance.create_entity(
                entity_type="modality_data",
                metadata={
                    "modality_name": name,
                    "adapter": adapter,
                    "shape": adata.shape,
                },
            )

            # Create IR for data loading operation
            source_path = (
                str(source) if not _is_anndata_instance(source) else "AnnData object"
            )
            if isinstance(source, (str, Path)):
                # A file load goes through the adapter, which changes the data
                # on load; replay it through the same shared adapter factory.
                suffix = Path(source).suffix.lower().lstrip(".")
                loading_ir = create_adapter_loading_ir(
                    adapter_name=adapter,
                    adapter_kwargs=kwargs,
                    input_param_name="input_data",
                    description=f"Load {name} data from {source_path} using the {adapter} adapter",
                    file_format=suffix or None,
                )
            else:
                loading_ir = create_data_loading_ir(
                    input_param_name="input_data",
                    description=f"Load {name} data from {source_path}",
                )

            # Log data loading operation with IR
            self.log_tool_usage(
                tool_name="load_dataset",
                parameters={
                    "name": name,
                    "source": source_path,
                    "adapter": adapter,
                    **kwargs,
                },
                description=f"Loaded modality '{name}' using {adapter} adapter",
                ir=loading_ir,
            )

            # Add provenance to AnnData
            adata = self.provenance.add_to_anndata(adata)

        logger.info(
            f"Loaded modality '{name}': {adata.shape} using adapter '{adapter}'"
        )

        # Notify listeners
        if self.on_modality_loaded is not None:
            try:
                self.on_modality_loaded(name, adata)
            except (
                Exception
            ):  # nosec B110 # Optional notification failure must not invalidate loaded data.
                pass  # Don't let callback errors break data loading

        return adata

    def store_modality(
        self,
        name: str,
        adata: "AnnData",
        parent_name: Optional[str] = None,
        step: Optional[str] = None,
        step_summary: Optional[str] = None,
        auto_save: bool = False,
    ) -> str:
        """
        Store a modality with automatic lineage tracking.

        This is the recommended method for storing processed modalities.
        It automatically:
        1. Extracts base_name from the modality name or parent
        2. Computes version based on parent modality
        3. Infers processing_step from the suffix
        4. Attaches lineage metadata to adata.uns['lineage']
        5. Marks modality as dirty for auto-save

        Args:
            name: Name for the modality (e.g., "geo_gse12345_clustered")
            adata: AnnData object to store
            parent_name: Name of parent modality this was derived from
            step: Processing step (auto-inferred from name if not provided).
                  One of: raw, quality_assessed, filtered, normalized,
                  filtered_normalized, batch_corrected, reduced, clustered,
                  subclustered, annotated, markers, de_results, pseudobulk, custom
            step_summary: Human-readable summary (e.g., "Filtered 15% low-quality cells")
            auto_save: Whether to immediately save to disk

        Returns:
            str: The modality name (for chaining)

        Example:
            >>> # In agent tool:
            >>> adata_clustered, stats, ir = clustering_service.cluster(adata)
            >>> data_manager.store_modality(
            ...     name=f"{modality_name}_clustered",
            ...     adata=adata_clustered,
            ...     parent_name=modality_name,
            ...     step_summary=f"Clustered into {stats['n_clusters']} clusters"
            ... )
        """
        # The name becomes part of file names: it must be one plain name.
        check_modality_name(name)
        from lobster.core.lineage import (
            attach_lineage,
            create_lineage_metadata,
        )

        # Get parent AnnData if available
        parent_adata = None
        if parent_name and parent_name in self.modalities:
            parent_adata = self.modalities[parent_name]

        # Create lineage metadata (step is any string — omics-agnostic)
        lineage = create_lineage_metadata(
            modality_name=name,
            parent_modality=parent_name,
            step=step,
            step_summary=step_summary,
            parent_adata=parent_adata,
        )

        # Attach lineage to AnnData
        attach_lineage(adata, lineage)

        # Store in modalities dict + backend if attached
        with self._state_lock:
            self.modalities[name] = adata
            if hasattr(self.modalities, "move_to_end"):
                self.modalities.move_to_end(name)
            self._evict_lru_if_needed()

        if self._modality_backend is not None:
            try:
                self._modality_backend.ingest_anndata(name, adata)
            except Exception as e:
                logger.error(f"Backend ingest failed for '{name}': {e}")
                self._modality_dirty.add(name)
        else:
            self._modality_dirty.add(name)

        # Log lineage creation
        logger.debug(
            f"Stored modality '{name}' with lineage: "
            f"base={lineage.base_name}, v{lineage.version}, "
            f"step={lineage.processing_step}"
        )

        # Auto-save if requested (legacy path, no-op when backend handles it)
        if auto_save and self._modality_backend is None:
            self.save_modality(name, f"{name}.h5ad")

        return name

    def save_modality(
        self, name: str, path: Union[str, Path], backend: Optional[str] = None, **kwargs
    ) -> str:
        """
        Save a modality using specified backend.

        Path contract: relative paths are resolved to absolute against
        ``data_dir`` for provenance/logging, then converted back to a
        backend-relative path via two-step resolution:

        1. ``relative_to(data_dir)``  — e.g. ``x.h5ad``
        2. ``relative_to(workspace_path)`` — e.g. ``x.h5ad`` or ``subdir/x.h5ad``
        3. Keep absolute (fallback for ``/tmp/`` paths outside workspace)

        Backends receive the relative path and resolve it against their
        own base_path. The return value is always the absolute resolved
        path (for provenance and caller use).

        Args:
            name: Name of modality to save
            path: Destination path (relative to data_dir, or absolute)
            backend: Backend to use (default: default_backend)
            **kwargs: Additional parameters passed to backend

        Returns:
            str: Absolute path where data was saved

        Raises:
            ValueError: If modality or backend not found
            ManagedPathError: The destination (as spelled, or where the backend
                writes it) or its backup is inside the managed figure folder
        """
        # The backup name planned by auto-save (or, for a MuData save without one,
        # read below before anything changes) reaches the backend only: it is not a
        # save parameter, so it is not recorded with the save.
        backup_stamp = kwargs.pop("backup_stamp", None)
        with self._measure_step("dm:save_modality"):
            if name not in self.modalities:
                raise ValueError(f"Modality '{name}' not loaded")
            # The destination is built from caller input (often a modality name):
            # never write into the managed figure folder (raises ManagedPathError).
            self._assert_save_allowed(path)

            adata = self.modalities[name]

            if self._modality_backend is None:
                # Materializing (below) can spill other modalities: first check where
                # the selected native backend writes (checked again before the save).
                selected = self.backends.get(backend or self.default_backend)
                if selected is not None:
                    requested = Path(path)
                    if not requested.is_absolute():
                        requested = self.data_dir / path
                    selected_path = self._backend_path(requested)
                    self._assert_backend_destination_allowed(selected, selected_path)
                    if _is_mudata_backend(selected):
                        # One backup name per save, fixed before anything changes:
                        # the backup made below uses the name checked here.
                        if backup_stamp is None:
                            backup_stamp = selected.backup_stamp()
                        self._assert_planned_backup_allowed(
                            selected,
                            selected_path,
                            backup_stamp,
                            loading=name if getattr(adata, "isbacked", False) else None,
                        )

            # Backed-mode AnnData must be materialized before save
            if getattr(adata, "isbacked", False):
                adata = self.materialize_modality(name)

            # When modality backend is attached, persist through it first
            if self._modality_backend is not None:
                try:
                    self._modality_backend.ingest_anndata(name, adata)
                    self._modality_dirty.discard(name)
                except Exception as e:
                    logger.error(f"Backend save failed for '{name}': {e}")
                    # Do not export stale backend state after a failed ingest.
                    raise

                # Still write h5ad file at requested path for caller
                resolved_path = Path(path)
                if not resolved_path.is_absolute():
                    resolved_path = self.data_dir / path
                self._modality_backend.export_h5ad(name, resolved_path)
                return str(resolved_path)

            backend_name = backend or self.default_backend
            if backend_name not in self.backends:
                raise ValueError(f"Backend '{backend_name}' not registered")

            backend_instance = self.backends[backend_name]

            # Resolve to absolute path for filesystem ops and provenance
            resolved_path = Path(path)
            if not resolved_path.is_absolute():
                resolved_path = self.data_dir / path

            # Compute backend-relative path so remote backends (S3) get
            # clean keys instead of embedded absolute filesystem paths.
            # Local backends re-resolve via base_path in _resolve_path().
            backend_path = self._backend_path(resolved_path)
            self._assert_backend_destination_allowed(backend_instance, backend_path)

            # Validate data for H5AD serialization (optional pre-save check)
            if hasattr(adata, "uns"):
                validation_issues = validate_for_h5ad(adata.uns, path="adata.uns")
                if validation_issues:
                    # The count only: the issues can quote names and values (DEBUG).
                    logger.warning(
                        "Pre-save validation found %d potential serialization "
                        "issues. These will be sanitized automatically during save.",
                        len(validation_issues),
                    )
                    logger.debug(
                        f"Pre-save validation issues in modality '{name}':\n"
                        + "\n".join(f"  - {issue}" for issue in validation_issues[:5])
                    )
                    if len(validation_issues) > 5:
                        logger.debug(
                            f"  ... and {len(validation_issues) - 5} more issues. "
                            f"Set logging to DEBUG for full list."
                        )

            # Save data — pass relative path to backend
            if backup_stamp is None:
                backend_instance.save(adata, backend_path, **kwargs)
            else:
                backend_instance.save(
                    adata, backend_path, backup_stamp=backup_stamp, **kwargs
                )
            path = resolved_path

            # Log provenance
            if self.provenance:
                self.provenance.create_entity(
                    entity_type="modality_data",
                    uri=path,
                    metadata={"modality_name": name, "shape": adata.shape},
                )

                # Create IR for data saving operation
                saving_ir = create_data_saving_ir(
                    output_prefix_param="output_prefix",
                    filename_suffix=Path(path).stem,
                    description=f"Save {name} data to {path}",
                    compression=kwargs.get("compression", "gzip"),
                )

                # Log data saving operation with IR
                self.log_tool_usage(
                    tool_name="save_dataset",
                    parameters={
                        "name": name,
                        "path": str(path),
                        "backend": backend_name,
                        **kwargs,
                    },
                    description=(
                        f"Saved modality '{name}' to {path} using {backend_name} backend"
                    ),
                    ir=saving_ir,
                )

            logger.info(
                f"Saved modality '{name}' to {path} using backend '{backend_name}'"
            )
            return str(path)

    def set_modality_backend(
        self,
        backend: Any,
        cache_cap: int = 8,
    ) -> None:
        """Attach a WorkspaceModalityBackend for lazy modality loading.

        When set, get_modality() delegates to the backend with LRU caching.
        When None (default), the existing dict path is used.
        Feature-flagged: only call this when LOBSTER_MODALITY_BACKEND=1.
        """
        from collections import OrderedDict

        self._modality_backend = backend
        self._modality_cache_cap = cache_cap

        existing = dict(self.modalities)
        self.modalities = OrderedDict()

        for name, adata in existing.items():
            if name in self._modality_dirty:
                try:
                    backend.ingest_anndata(name, adata)
                    self._modality_dirty.discard(name)
                    logger.info(f"Migrated dirty modality '{name}' to backend")
                except Exception as e:
                    logger.error(f"Failed to migrate '{name}': {e}")
                    self.modalities[name] = adata
                    continue
            _close_backed_handle(adata)

        logger.info(
            f"Modality backend set: {type(backend).__name__}, LRU cap={cache_cap}"
        )

    def _evict_lru_if_needed(self) -> None:
        """Evict least-recently-used modalities to stay within limits.

        Two independent limits apply:

        * When a ``WorkspaceModalityBackend`` is attached, a **count** cap
          (``_modality_cache_cap``); dirty modalities are flushed to the
          backend before eviction to prevent data loss.
        * Always, a **byte budget** (``_modality_memory_budget``) enforced by
          spilling to h5ad. This bounds accumulated modality memory even when a
          count cap alone would allow a few large entries to exhaust memory.

        Must be called with ``_state_lock`` held.
        """
        if self._modality_backend is not None:
            while len(self.modalities) > self._modality_cache_cap:
                evicted_name, evicted_adata = next(iter(self.modalities.items()))
                if evicted_name in self._modality_dirty:
                    try:
                        self._modality_backend.ingest_anndata(
                            evicted_name, evicted_adata
                        )
                        self._modality_dirty.discard(evicted_name)
                        logger.info(
                            f"Flushed dirty modality '{evicted_name}' before eviction"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to flush dirty modality '{evicted_name}' "
                            f"before eviction: {e}. Keeping in cache."
                        )
                        return
                _close_backed_handle(evicted_adata)
                del self.modalities[evicted_name]
                self._modality_sources.pop(evicted_name, None)
                logger.info(f"LRU evicted modality '{evicted_name}'")

        self._enforce_memory_budget()

    def _enforce_memory_budget(self) -> None:
        """Spill least-recently-used modalities until under the byte budget.

        The most-recently-used entry is never spilled: it is the one the caller
        just stored or fetched and is about to use, so evicting it would
        guarantee an immediate reload.  Consequently a single modality larger
        than the whole budget is kept — the budget bounds accumulation, not
        one oversized object.

        Must be called with ``_state_lock`` held.
        """
        if self._modality_memory_budget <= 0:
            return

        total = self._resident_bytes()
        if total <= self._modality_memory_budget:
            return

        # Oldest first; stop before the most-recently-used entry.
        for name in list(self.modalities.keys())[:-1]:
            if total <= self._modality_memory_budget:
                break
            freed = self._spill_modality(name)
            if freed is None:
                # Spill failed; stop rather than spin on the same entry.
                break
            total -= freed

        if total > self._modality_memory_budget:
            logger.warning(
                f"Modality registry still holds ~{total / 1024**3:.2f} GiB after "
                f"eviction, above the "
                f"{self._modality_memory_budget / 1024**3:.2f} GiB budget "
                f"({len(self.modalities)} modality/-ies resident). The most "
                f"recently used modality is never spilled."
            )

    def _resident_bytes(self) -> int:
        """Estimated total resident bytes held by the modality registry."""
        return sum(estimate_anndata_bytes(a) for a in self.modalities.values())

    def _spill_modality(self, name: str) -> Optional[int]:
        """Write one modality to the spill directory and drop it from memory.

        Returns the estimated bytes freed, or ``None`` if the modality could
        not be spilled (in which case it is left resident and intact).
        """
        adata = self.modalities.get(name)
        if adata is None:
            return None

        freed = estimate_anndata_bytes(adata)

        # Already on disk and not modified since load: reuse the source file
        # instead of writing a second copy.
        source = self._modality_sources.get(name)
        if (
            source is not None
            and name not in self._modality_dirty
            and Path(source).exists()
        ):
            spill_path = Path(source)
        else:
            try:
                spill_path = self._modality_spill_dir / f"{_safe_filename(name)}.h5ad"
                # The folder and the file itself, before the folder is created.
                _assert_dirs_allowed(self.workspace_path, self._modality_spill_dir)
                self._assert_save_allowed(spill_path)
                self._modality_spill_dir.mkdir(parents=True, exist_ok=True)
                to_write = adata
                if getattr(adata, "isbacked", False):
                    to_write = adata.to_memory()
                validate_for_h5ad(to_write)
                self._assert_save_allowed(spill_path)
                to_write.write_h5ad(spill_path)
            except ManagedPathError:
                logger.warning(_SPILL_REFUSED_LOG)
                return None
            except Exception as e:
                logger.error(
                    f"Failed to spill modality '{name}' to disk: {e}. "
                    f"Keeping it resident."
                )
                return None

        _close_backed_handle(adata)
        del self.modalities[name]
        self._modality_dirty.discard(name)
        self._modality_spilled[name] = spill_path
        logger.info(
            f"Spilled modality '{name}' (~{freed / 1024**3:.2f} GiB) to "
            f"{spill_path.name}; it will be reloaded on next access"
        )
        return freed

    def _restore_spilled_modality(self, name: str) -> Optional["AnnData"]:
        """Reload a previously spilled modality from disk.

        Must be called with ``_state_lock`` held.
        """
        spill_path = self._modality_spilled.get(name)
        if spill_path is None:
            return None
        path = Path(spill_path)
        if not path.exists():
            logger.error(
                f"Spill file for modality '{name}' is missing at {path}; "
                f"the modality cannot be restored"
            )
            self._modality_spilled.pop(name, None)
            return None

        adata = _read_h5ad_smart(path, backed_mode_enabled=self.backed_mode)
        self.modalities[name] = adata
        if hasattr(self.modalities, "move_to_end"):
            self.modalities.move_to_end(name)
        self._modality_spilled.pop(name, None)
        logger.info(f"Restored spilled modality '{name}' from {path.name}")
        # Restoring may push us back over budget; spill something older.
        self._enforce_memory_budget()
        return self.modalities.get(name, adata)

    def get_modality(self, name: str) -> "AnnData":
        """Get a specific modality.

        When a WorkspaceModalityBackend is attached, materializes from
        the backend on cache miss and applies LRU eviction.
        """
        with self._state_lock:
            if name in self.modalities:
                if hasattr(self.modalities, "move_to_end"):
                    self.modalities.move_to_end(name)
                adata = self.modalities[name]
                if getattr(adata, "isbacked", False):
                    backed = adata
                    adata = backed.to_memory()
                    _close_backed_handle(backed)
                    self.modalities[name] = adata
                    self._modality_dirty.add(name)
                    logger.info(f"Auto-materialized backed modality '{name}'")
                return adata

            # A modality spilled to disk to stay within the memory
            # budget is still a live modality — reload it rather than 404.
            if name in self._modality_spilled:
                restored = self._restore_spilled_modality(name)
                if restored is not None:
                    return restored

            if self._modality_backend is not None:
                if not self._modality_backend.exists(name):
                    raise ValueError(f"Modality '{name}' not found")
                adata = self._modality_backend.materialize_anndata(name)
                self.modalities[name] = adata
                if hasattr(self.modalities, "move_to_end"):
                    self.modalities.move_to_end(name)
                self._evict_lru_if_needed()
                return adata

            raise ValueError(f"Modality '{name}' not found")

    def is_backed(self, name: str) -> bool:
        """Return True if modality is loaded in HDF5 backed mode."""
        with self._state_lock:
            if name not in self.modalities:
                return False
            return getattr(self.modalities[name], "isbacked", False)

    def materialize_modality(self, name: str) -> "AnnData":
        """Convert a backed-mode modality to fully in-memory.

        Safe to call on already-materialized modalities (no-op).
        Must be called before any operation that writes to .X.
        On .to_memory() failure, HDF5 handle is left open for retry.
        """
        with self._state_lock:
            if name not in self.modalities:
                if name in self._modality_spilled:
                    self._restore_spilled_modality(name)
                if name not in self.modalities:
                    raise ValueError(f"Modality '{name}' not found")
            adata = self.modalities[name]
            if getattr(adata, "isbacked", False):
                logger.info(f"Materializing backed modality '{name}' to memory")
                backed = adata
                adata = backed.to_memory()
                _close_backed_handle(backed)
                self.modalities[name] = adata
                self._modality_dirty.add(name)
                # Materializing grows resident bytes; re-check the budget.
                self._enforce_memory_budget()
            return adata

    def ensure_in_memory(self, name: str) -> "AnnData":
        """Get modality, materialize if backed, and mark dirty.

        Callers of ensure_in_memory() declare write intent — the modality
        is auto-marked dirty so LRU eviction flushes it to backend.
        Handles cold (backend-only) modalities by loading them first.
        Idempotent. Tools that mutate should call this, not get_modality().
        """
        with self._state_lock:
            if name not in self.modalities:
                # A spilled modality is still live — restore it, or a
                # mutating tool would see a spurious "not found".
                if name in self._modality_spilled:
                    self._restore_spilled_modality(name)
                if name in self.modalities:
                    pass
                elif (
                    self._modality_backend is not None
                    and self._modality_backend.exists(name)
                ):
                    adata = self._modality_backend.materialize_anndata(name)
                    self.modalities[name] = adata
                    if hasattr(self.modalities, "move_to_end"):
                        self.modalities.move_to_end(name)
                    self._evict_lru_if_needed()
                else:
                    raise ValueError(f"Modality '{name}' not found")
            adata = self.modalities[name]
            if getattr(adata, "isbacked", False):
                logger.info(f"Materializing backed modality '{name}' to memory")
                backed = adata
                try:
                    adata = backed.to_memory()
                except Exception:
                    raise
                finally:
                    _close_backed_handle(backed)
                self.modalities[name] = adata
            self._modality_dirty.add(name)
            return adata

    def list_modalities(self) -> List[str]:
        """List all available modalities (cache + spilled + backend union)."""
        with self._state_lock:
            names = set(self.modalities.keys())
            names.update(self._modality_spilled.keys())
            if self._modality_backend is not None:
                try:
                    for rec in self._modality_backend.list_modalities():
                        names.add(rec.name)
                except Exception as e:
                    logger.warning(f"Failed to list backend modalities: {e}")
            return sorted(names)

    def list_modality_records(self) -> List[Dict[str, Any]]:
        """List modalities with rich metadata including data_status.

        Returns records for all modalities (hot=cached, cold=backend-only).
        """
        with self._state_lock:
            records: List[Dict[str, Any]] = []
            seen: set = set()

            for name, adata in self.modalities.items():
                seen.add(name)
                is_dirty = name in self._modality_dirty
                records.append(
                    {
                        "name": name,
                        "n_obs": getattr(adata, "n_obs", 0),
                        "n_vars": getattr(adata, "n_vars", 0),
                        "data_status": "hot",
                        "is_dirty": is_dirty,
                        "is_backed": getattr(adata, "isbacked", False),
                    }
                )

            # Spilled modalities are cold but still present.
            for name, spill_path in self._modality_spilled.items():
                if name in seen:
                    continue
                seen.add(name)
                record: Dict[str, Any] = {
                    "name": name,
                    "n_obs": 0,
                    "n_vars": 0,
                    "data_status": "spilled",
                    "is_dirty": False,
                    "is_backed": False,
                }
                try:
                    record["storage_size_bytes"] = Path(spill_path).stat().st_size
                except OSError:
                    pass
                records.append(record)

            if self._modality_backend is not None:
                try:
                    for rec in self._modality_backend.list_modalities():
                        if rec.name not in seen:
                            records.append(
                                {
                                    "name": rec.name,
                                    "n_obs": rec.n_obs,
                                    "n_vars": rec.n_vars,
                                    "data_status": "cold",
                                    "is_dirty": False,
                                    "is_backed": False,
                                    "storage_size_bytes": rec.storage_size_bytes,
                                }
                            )
                except Exception as e:
                    logger.warning(f"Failed to list backend modalities: {e}")

            return records

    def list_modalities_with_lineage(self) -> List[Dict[str, Any]]:
        """
        List all modalities with lineage metadata for frontend display.

        Returns list sorted by base_name, then version for timeline display.

        Returns:
            List of dicts with modality info and lineage metadata:
            - name: Modality name
            - base_name: Base dataset name
            - version: Version number
            - processing_step: Processing step label
            - parent_modality: Parent modality name or None
            - step_summary: Human-readable summary
            - created_at: ISO timestamp
            - shape: Dict with n_obs and n_vars
        """
        from lobster.core.lineage import LINEAGE_KEY

        result = []
        for name, adata in self.modalities.items():
            lineage = adata.uns.get(LINEAGE_KEY, {})
            result.append(
                {
                    "name": name,
                    "base_name": lineage.get("base_name", name),
                    "version": int(lineage.get("version", 1)),
                    "processing_step": lineage.get("processing_step", "raw"),
                    "parent_modality": lineage.get("parent_modality"),
                    "step_summary": lineage.get("step_summary"),
                    "created_at": lineage.get("created_at"),
                    "shape": {"n_obs": adata.n_obs, "n_vars": adata.n_vars},
                }
            )

        # Sort by base_name, then version
        return sorted(result, key=lambda x: (x["base_name"], x["version"]))

    def get_modalities_by_base_name(self, base_name: str) -> List[str]:
        """
        Get all modalities derived from a base dataset.

        Args:
            base_name: Base dataset name to filter by

        Returns:
            List of modality names with matching base_name, sorted by version
        """
        from lobster.core.lineage import LINEAGE_KEY

        result = []
        for name, adata in self.modalities.items():
            lineage = adata.uns.get(LINEAGE_KEY, {})
            if lineage.get("base_name") == base_name:
                result.append((name, int(lineage.get("version", 0))))

        # Sort by version and return just names
        return [name for name, _ in sorted(result, key=lambda x: x[1])]

    def get_lineage_tree(self, modality_name: str) -> Dict[str, Any]:
        """
        Get full lineage tree for frontend visualization.

        Args:
            modality_name: Name of modality to get tree for

        Returns:
            Nested dict structure with:
            - modality: Modality name
            - lineage: Lineage metadata dict
            - shape: Shape info (n_obs, n_vars)
            - parent: Parent's lineage tree (recursive) or None
            - children: List of child modality summaries
        """
        from lobster.core.lineage import LINEAGE_KEY

        if modality_name not in self.modalities:
            return {}

        adata = self.modalities[modality_name]
        lineage = adata.uns.get(LINEAGE_KEY, {})

        result: Dict[str, Any] = {
            "modality": modality_name,
            "lineage": lineage,
            "shape": {"n_obs": adata.n_obs, "n_vars": adata.n_vars},
        }

        # Get parent recursively
        parent_name = lineage.get("parent_modality")
        if parent_name and parent_name in self.modalities:
            result["parent"] = self.get_lineage_tree(parent_name)
        else:
            result["parent"] = None

        # Get children
        result["children"] = []
        for name, child_adata in self.modalities.items():
            child_lineage = child_adata.uns.get(LINEAGE_KEY, {})
            if child_lineage.get("parent_modality") == modality_name:
                result["children"].append(
                    {
                        "modality": name,
                        "lineage": child_lineage,
                        "shape": {
                            "n_obs": child_adata.n_obs,
                            "n_vars": child_adata.n_vars,
                        },
                    }
                )

        return result

    def remove_modality(self, name: str) -> None:
        """
        Remove a modality from memory.

        Args:
            name: Name of modality to remove

        Raises:
            ValueError: If modality not found
        """
        with self._state_lock:
            in_cache = name in self.modalities
            # A spilled modality is still present, so removing it must
            # succeed rather than raise, and must delete the spill file.
            spill_path = self._modality_spilled.pop(name, None)
            in_backend = (
                self._modality_backend is not None
                and self._modality_backend.exists(name)
            )
            if not in_cache and not in_backend and spill_path is None:
                raise ValueError(f"Modality '{name}' not found")
            if in_backend:
                self._modality_backend.remove(name)
            if spill_path is not None:
                try:
                    path = Path(spill_path)
                    # Only delete files we wrote; never a user's source data.
                    if path.parent == self._modality_spill_dir and path.exists():
                        path.unlink()
                except OSError as e:
                    logger.warning(f"Could not delete spill file {spill_path}: {e}")
            if in_cache:
                _close_backed_handle(self.modalities[name])
                del self.modalities[name]
                self._modality_dirty.discard(name)
                self._modality_sources.pop(name, None)
        logger.info(f"Removed modality '{name}'")

    def to_mudata(self, modalities: Optional[List[str]] = None) -> Any:
        """
        Convert modalities to MuData object.

        Args:
            modalities: List of modality names to include (default: all)

        Returns:
            mudata.MuData: MuData object containing specified modalities

        Raises:
            ImportError: If MuData is not available
            ValueError: If no modalities are loaded
        """
        if not MUDATA_AVAILABLE:
            from lobster.core.component_registry import get_install_command

            cmd = get_install_command("mudata")
            raise ImportError(f"MuData is not available. Install with: {cmd}")

        if not self.modalities:
            raise ValueError("No modalities loaded")

        # Use specified modalities or all available
        modality_names = modalities or list(self.modalities.keys())

        # Check that all requested modalities exist
        missing = [name for name in modality_names if name not in self.modalities]
        if missing:
            raise ValueError(f"Modalities not found: {missing}")

        # Create modality dictionary
        modality_dict = {name: self.modalities[name] for name in modality_names}

        # Create MuData object
        mdata = mudata.MuData(modality_dict)

        # Add provenance if enabled
        if self.provenance:
            mdata = self.provenance.add_to_anndata(mdata)

        logger.info(
            f"Created MuData with {len(modality_dict)} modalities: {list(modality_dict.keys())}"
        )
        return mdata

    def _assert_save_allowed(self, path: Union[str, Path]) -> None:
        """Raise ManagedPathError when a save to ``path`` (relative to data_dir, or
        absolute) would land in the managed figure folder. Checked on the unresolved
        spelling, lexically and resolved, before anything is written."""
        from lobster.core.figures.export_guard import assert_export_allowed

        target = Path(path)
        if not target.is_absolute():
            target = self.data_dir / path
            # A relative save stays inside the workspace (a '..' or a link leading
            # out raises WorkspaceExportError); an absolute path is the caller's own.
            assert_inside_workspace(self.workspace_path, target)
        assert_export_allowed(self.workspace_path, target)

    def _backend_path(self, resolved_path: Path) -> Union[str, Path]:
        """The path handed to a backend: relative to data_dir, else to the
        workspace, else absolute (a local backend re-roots a relative one)."""
        try:
            return resolved_path.relative_to(self.data_dir)
        except ValueError:
            try:
                return resolved_path.relative_to(self.workspace_path)
            except ValueError:
                return resolved_path

    def _assert_backend_destination_allowed(
        self, backend: Any, backend_path: Union[str, Path]
    ) -> None:
        """Raise ManagedPathError when the local file a native backend writes for
        ``backend_path`` (its own resolution: a re-rooted path lands under data/) or
        a folder it creates for it is inside the managed figure folder."""
        native = tuple(
            cls
            for cls in (
                H5ADBackend,
                MuDataBackend if MUDATA_BACKEND_AVAILABLE else None,
            )
            if isinstance(cls, type)  # None when unavailable (or replaced in tests)
        )
        if not isinstance(backend, native):
            return  # other registered backends are not classified here
        is_s3 = getattr(backend, "_is_s3_path", None)
        if is_s3 is not None and is_s3(backend_path):
            return  # handed to the remote writer before any local write
        destination = Path(backend._resolve_path(backend_path))
        if not destination.is_absolute():
            destination = Path.cwd() / destination
        self._assert_save_allowed(destination)
        _assert_dirs_allowed(self.workspace_path, destination.parent)

    def _assert_planned_backup_allowed(
        self,
        backend: Any,
        backend_path: Union[str, Path],
        stamp: str,
        *,
        loading: Optional[str],
    ) -> None:
        """Raise ManagedPathError when the backup the MuData ``backend`` may make for
        ``backend_path`` under ``stamp`` is refused (no effect). ``loading`` names the
        backed modality the save loads first (None when nothing is loaded).

        The backup name is computed exactly as the writer computes it; when that
        fails, the writer logs a failed backup and saves, so nothing is checked here.
        Otherwise the backup is checked unless the writer certainly makes none (see
        :meth:`_backup_may_be_made`). The writer checks it again before copying.
        """
        is_s3 = getattr(backend, "_is_s3_path", None)
        if is_s3 is not None and is_s3(backend_path):
            return  # handed to the remote writer before any local write
        try:
            destination = Path(backend._resolve_path(backend_path))
            source = backend.backup_source(destination)
            backup = backend.backup_sibling(source, stamp)
        except Exception:
            return  # the writer makes no backup it cannot name (logged, as before)
        if self._backup_may_be_made(destination, source, loading):
            backend._assert_backup_allowed(backup)

    def _backup_may_be_made(
        self, destination: Path, source: Path, loading: Optional[str]
    ) -> bool:
        """False only when the writer certainly makes no backup (no effect).

        The writer makes one when, at write time, both the destination and the file
        the backup copies exist. A path that does not exist now may exist then when
        its spelling only lacks a missing folder (the writer, or the spill, creates
        folders) or, for a save that loads backed data, when that load can create it
        (:meth:`_paths_a_load_can_create`). Anything that cannot be evaluated counts
        as a backup that may be made.
        """
        try:
            created = None

            def may_exist(path: Path) -> bool:
                nonlocal created
                if path.exists() or path.resolve().exists():
                    return True
                if loading is None:
                    return False
                if created is None:
                    created = self._paths_a_load_can_create(loading)
                return path.resolve() in created

            return may_exist(destination) and may_exist(source)
        except Exception:
            return True

    def _paths_a_load_can_create(self, loading: str) -> frozenset:
        """The paths loading the backed modality ``loading`` can create (resolved,
        no effect).

        The memory budget can spill resident modalities, writing a spill file and
        creating the spill folder and its missing parents. The most recently used
        entry is never spilled, and a modality unchanged since it was read from a
        file that still exists reuses that file (nothing is written). The modality
        being loaded is marked modified first, so it never reuses its file.
        """
        if self._modality_memory_budget <= 0:
            return frozenset()
        writes = []
        for name in list(self.modalities.keys())[:-1]:
            source = self._modality_sources.get(name)
            reuses = (
                name != loading
                and source is not None
                and name not in self._modality_dirty
                and Path(source).exists()
            )
            if not reuses:
                writes.append(name)
        if not writes:
            return frozenset()
        folder = Path(self._modality_spill_dir)
        if not folder.is_absolute():
            folder = Path.cwd() / folder
        created = {prefix.resolve() for prefix in (folder, *folder.parents)}
        created.update(
            (folder / f"{_safe_filename(name)}.h5ad").resolve() for name in writes
        )
        return frozenset(created)

    def save_mudata(
        self, path: Union[str, Path], modalities: Optional[List[str]] = None, **kwargs
    ) -> str:
        """
        Save modalities as MuData file.

        Uses the same two-step path resolution as ``save_modality()``:
        backends receive a relative path, return value is absolute.

        Args:
            path: Destination path (relative to data_dir, or absolute)
            modalities: List of modality names to include (default: all)
            **kwargs: Additional parameters passed to MuData backend

        Returns:
            str: Absolute path where data was saved

        Raises:
            ValueError: If MuData backend not available
        """
        if "mudata" not in self.backends:
            raise ValueError("MuData backend not available")
        self._assert_save_allowed(path)  # never into the managed figure folder

        # Create MuData object
        mdata = self.to_mudata(modalities=modalities)

        # Resolve to absolute path for provenance/logging
        resolved_path = Path(path)
        if not resolved_path.is_absolute():
            resolved_path = self.data_dir / path

        # Compute backend-relative path (same logic as save_modality)
        backend_path = self._backend_path(resolved_path)

        # Save using MuData backend
        mudata_backend = self.backends["mudata"]
        self._assert_backend_destination_allowed(mudata_backend, backend_path)
        mudata_backend.save(mdata, backend_path, **kwargs)

        logger.info(f"Saved MuData to {resolved_path}")
        return str(resolved_path)

    def _match_modality_to_adapter(self, modality_name: str) -> Optional[str]:
        """
        Smart matching of modality names to adapter names.

        This method uses a hierarchical matching strategy to identify the correct
        adapter for a given modality name, handling various naming patterns.

        Args:
            modality_name: Name of the modality to match

        Returns:
            Optional[str]: Matching adapter name, or None if no match found
        """
        modality_lower = modality_name.lower()

        # Define keyword mappings for each adapter type
        adapter_keywords = {
            "transcriptomics_single_cell": {
                "primary": ["single_cell", "single-cell", "sc_", "scrna", "10x"],
                "secondary": ["transcriptom", "rna", "gene_expression"],
                "exclude": ["bulk"],
            },
            "transcriptomics_bulk": {
                "primary": ["bulk"],
                "secondary": ["transcriptom", "rna", "gene_expression"],
                "exclude": ["single_cell", "single-cell", "sc_", "scrna"],
            },
            "proteomics_ms": {
                "primary": ["ms", "mass_spec", "mass-spec", "mass_spectrometry"],
                "secondary": ["proteomic", "protein"],
                "exclude": ["affinity", "antibody"],
            },
            "proteomics_affinity": {
                "primary": ["affinity", "antibody", "immunoassay", "elisa", "western"],
                "secondary": ["proteomic", "protein"],
                "exclude": ["ms", "mass_spec"],
            },
        }

        # Score each adapter based on keyword matches
        adapter_scores = {}

        for adapter_name, keywords in adapter_keywords.items():
            score = 0

            # Check exclusion keywords first - if any match, skip this adapter
            if any(
                exclude in modality_lower for exclude in keywords.get("exclude", [])
            ):
                continue

            # Primary keywords have higher weight
            for primary in keywords.get("primary", []):
                if primary in modality_lower:
                    score += 10

            # Secondary keywords have lower weight
            for secondary in keywords.get("secondary", []):
                if secondary in modality_lower:
                    score += 5

            # Store score if > 0
            if score > 0:
                adapter_scores[adapter_name] = score

        # Return adapter with highest score, or None if no matches
        if adapter_scores:
            return max(adapter_scores.items(), key=lambda x: x[1])[0]

        # Fallback: try to infer from modality name patterns
        if "geo" in modality_lower or "gse" in modality_lower:
            # GEO datasets are usually transcriptomics
            if any(keyword in modality_lower for keyword in ["single", "sc", "10x"]):
                return "transcriptomics_single_cell"
            else:
                return "transcriptomics_bulk"

        return None

    def get_quality_metrics(self, modality: Optional[str] = None) -> Dict[str, Any]:
        """
        Get quality metrics for modalities.

        Args:
            modality: Specific modality name (default: all modalities)

        Returns:
            Dict[str, Any]: Quality metrics
        """
        if modality:
            if modality not in self.modalities:
                raise ValueError(f"Modality '{modality}' not found")

            # Use smart matching to find appropriate adapter
            matched_adapter_name = self._match_modality_to_adapter(modality)

            if matched_adapter_name and matched_adapter_name in self.adapters:
                adapter_instance = self.adapters[matched_adapter_name]
                logger.debug(
                    f"Matched modality '{modality}' to adapter '{matched_adapter_name}'"
                )
                return adapter_instance.get_quality_metrics(self.modalities[modality])
            else:
                # Use basic metrics if no specific adapter found
                logger.warning(
                    f"No specific adapter found for modality '{modality}', using basic metrics"
                )
                base_adapter = BaseAdapter()
                return base_adapter.get_quality_metrics(self.modalities[modality])

        else:
            # Return metrics for all modalities
            all_metrics = {}
            for mod_name in self.modalities:
                all_metrics[mod_name] = self.get_quality_metrics(mod_name)
            return all_metrics

    def get_workspace_status(self) -> Dict[str, Any]:
        """
        Get comprehensive workspace status.

        Returns:
            Dict[str, Any]: Workspace status information
        """
        status = {
            "workspace_path": str(self.workspace_path),
            "modalities_loaded": len(self.modalities),
            "modality_names": list(self.modalities.keys()),
            "registered_backends": list(self.backends.keys()),
            "registered_adapters": list(self.adapters.keys()),
            "default_backend": self.default_backend,
            "provenance_enabled": self.enable_provenance,
            "mudata_available": MUDATA_AVAILABLE,
            "directories": {
                "data": str(self.data_dir),
                "exports": str(self.exports_dir),
                "cache": str(self.cache_dir),
                "literature_cache": str(self.literature_cache_dir),
                "metadata": str(self.workspace_path / "metadata"),
                "notebooks": str(self.workspace_path / "notebooks"),
                "queues": str(self._queues_dir),
            },
        }

        # Add modality details
        if self.modalities:
            status["modality_details"] = {}
            for name, adata in self.modalities.items():
                status["modality_details"][name] = {
                    "shape": adata.shape,
                    "obs_columns": list(adata.obs.columns)[:10],  # First 10 columns
                    "var_columns": list(adata.var.columns)[:10],  # First 10 columns
                    "layers": layer_names(adata),
                    "obsm": list(adata.obsm.keys()) if adata.obsm else [],
                }

        # Add provenance info
        if self.provenance:
            status["provenance"] = {
                "n_activities": len(self.provenance.activities),
                "n_entities": len(self.provenance.entities),
                "n_agents": len(self.provenance.agents),
            }

        return status

    def validate_modalities(self, strict: bool = False) -> Dict[str, ValidationResult]:
        """
        Validate all loaded modalities.

        Args:
            strict: Whether to use strict validation

        Returns:
            Dict[str, ValidationResult]: Validation results for each modality
        """
        results = {}

        for name, adata in self.modalities.items():
            # Use smart matching to find appropriate adapter
            matched_adapter_name = self._match_modality_to_adapter(name)

            if matched_adapter_name and matched_adapter_name in self.adapters:
                adapter_instance = self.adapters[matched_adapter_name]
                logger.debug(
                    f"Validating modality '{name}' with adapter '{matched_adapter_name}'"
                )
                results[name] = adapter_instance.validate(adata, strict=strict)
            else:
                # Use basic validation if no specific adapter found
                logger.warning(
                    f"No specific adapter found for modality '{name}', using basic validation"
                )
                base_adapter = BaseAdapter()
                results[name] = base_adapter._validate_basic_structure(adata)

        return results

    def export_provenance(self, path: Union[str, Path]) -> str:
        """
        Export provenance information to file.

        Args:
            path: Export file path

        Returns:
            str: Path to exported file

        Raises:
            ValueError: If provenance tracking is disabled
        """
        if not self.provenance:
            raise ValueError("Provenance tracking is disabled")

        # Resolve path
        if not Path(path).is_absolute():
            path = self.exports_dir / path

        # Export provenance
        provenance_data = self.provenance.to_dict()

        import json

        with open(path, "w") as f:
            json.dump(provenance_data, f, indent=2, default=str)

        logger.info(f"Exported provenance to {path}")
        return str(path)

    def save_workspace(self, workspace_name: str) -> Path:
        """
        Save all current modalities to a named workspace directory.

        This method creates a dedicated directory for the workspace and saves
        all loaded modalities as individual .h5ad files. It also saves session
        metadata including the list of active modalities for later restoration.

        Args:
            workspace_name: Name for the workspace (will create a subdirectory)

        Returns:
            Path: Path to the workspace directory containing saved modalities

        Example:
            >>> dm.save_workspace("my_analysis_20250116")
            PosixPath('/path/.lobster_workspace/my_analysis_20250116')

        Note:
            - Uses thread-safe locking to prevent concurrent saves
            - Saves all modalities as {modality_name}.h5ad files
            - Creates session metadata for restoration via restore_session()
        """
        # Create workspace directory
        workspace_dir = self.workspace_path / workspace_name
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Save each modality
        saved_modalities = []
        for modality_name in self.modalities:
            try:
                save_path = workspace_dir / f"{modality_name}.h5ad"
                self.save_modality(modality_name, str(save_path))
                saved_modalities.append(modality_name)
                logger.debug(f"Saved modality '{modality_name}' to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save modality '{modality_name}': {e}")

        # Save session metadata
        session_metadata = {
            "workspace_name": workspace_name,
            "active_modalities": saved_modalities,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "n_modalities": len(saved_modalities),
        }

        metadata_path = workspace_dir / ".session.json"
        with open(metadata_path, "w") as f:
            json.dump(session_metadata, f, indent=2)

        logger.info(
            f"Saved workspace '{workspace_name}' with {len(saved_modalities)} modalities to {workspace_dir}"
        )
        return workspace_dir

    def load_workspace(self, workspace_name: str) -> Dict[str, Any]:
        """
        Load all modalities from a previously saved workspace directory.

        This is the symmetric counterpart to save_workspace(). It reads
        all .h5ad files from the named workspace directory and loads them
        as modalities.

        Args:
            workspace_name: Name of the workspace to load (must have been
                previously saved via save_workspace)

        Returns:
            Dict[str, Any]: Summary of loaded workspace including list of
                restored modalities and any errors

        Raises:
            FileNotFoundError: If the workspace directory does not exist

        Example:
            >>> dm.save_workspace("my_analysis")
            >>> dm.clear()
            >>> dm.load_workspace("my_analysis")
            {'workspace_name': 'my_analysis', 'loaded': [...], 'errors': []}
        """
        workspace_dir = self.workspace_path / workspace_name
        if not workspace_dir.exists():
            raise FileNotFoundError(
                f"Workspace '{workspace_name}' not found at {workspace_dir}"
            )

        _ensure_anndata()
        loaded = []
        errors = []

        # Load session metadata if available
        metadata_path = workspace_dir / ".session.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                session_metadata = json.load(f)
        else:
            session_metadata = {}

        # Load all h5ad files from the workspace directory
        h5ad_files = sorted(workspace_dir.glob("*.h5ad"))
        for h5ad_file in h5ad_files:
            modality_name = h5ad_file.stem
            try:
                adata = _read_h5ad_smart(
                    h5ad_file, backed_mode_enabled=self.backed_mode
                )
                with self._state_lock:
                    old = self.modalities.get(modality_name)
                    if old is not None:
                        _close_backed_handle(old)
                    self.modalities[modality_name] = adata
                self._modality_sources[modality_name] = h5ad_file
                loaded.append(modality_name)
                backed_tag = " [backed]" if getattr(adata, "isbacked", False) else ""
                logger.debug(
                    f"Loaded modality '{modality_name}' from {h5ad_file}{backed_tag}"
                )
            except HdfReadRefusal:
                raise
            except Exception as e:
                logger.error(f"Failed to load modality '{modality_name}': {e}")
                errors.append({"modality": modality_name, "error": str(e)})

        logger.info(
            f"Loaded workspace '{workspace_name}' with {len(loaded)} modalities"
        )

        return {
            "workspace_name": workspace_name,
            "loaded": loaded,
            "errors": errors,
            "session_metadata": session_metadata,
        }

    def clear_workspace(self, confirm: bool = False) -> None:
        """
        Clear all modalities and optionally workspace files.

        Args:
            confirm: Must be True to actually clear workspace

        Raises:
            ValueError: If confirm is not True
        """
        if not confirm:
            raise ValueError("Must set confirm=True to clear workspace")

        # Close any backed HDF5 handles before clearing
        for adata in self.modalities.values():
            _close_backed_handle(adata)

        # Close modality backend if attached
        if self._modality_backend is not None:
            self._modality_backend.close()

        # Clear modalities from memory
        self.modalities.clear()

        # Reset provenance (preserve session_dir for continued persistence)
        if self.provenance:
            self.provenance = ProvenanceTracker(session_dir=self._session_dir)

        logger.info("Cleared workspace")

    def clear(self) -> None:
        """
        Clear all modalities from memory without requiring confirmation.

        This is a convenience alias for clear_workspace() that doesn't require
        the confirm=True parameter. Use this for programmatic clearing in tests
        and scripts where confirmation prompts are not needed.

        Note:
            This method clears modalities and resets provenance but does not
            delete files from disk.

        Example:
            >>> dm.clear()  # Simple, no confirmation needed
        """
        # Close any backed HDF5 handles before clearing
        for adata in self.modalities.values():
            _close_backed_handle(adata)

        # Close modality backend if attached
        if self._modality_backend is not None:
            self._modality_backend.close()

        # Clear modalities from memory
        self.modalities.clear()

        # Reset provenance (preserve session_dir for continued persistence)
        if self.provenance:
            self.provenance = ProvenanceTracker(session_dir=self._session_dir)

        logger.info("Cleared workspace (via clear() alias)")

    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get information about registered backends.

        Returns:
            Dict[str, Any]: Backend information
        """
        info = {}
        for name, backend in self.backends.items():
            info[name] = backend.get_storage_info()
        return info

    def get_adapter_info(self) -> Dict[str, Any]:
        """
        Get information about registered adapters.

        Returns:
            Dict[str, Any]: Adapter information
        """
        info = {}
        for name, adapter in self.adapters.items():
            info[name] = {
                "modality_name": adapter.get_modality_name(),
                "supported_formats": adapter.get_supported_formats(),
                "schema": adapter.get_schema(),
            }
        return info

    def attach_session_dir(self, session_dir: Union[str, Path]) -> Optional[Path]:
        """
        Point provenance persistence at a session directory after construction.

        ``DataManagerV2(workspace_path=...)`` without ``session_dir`` disables provenance
        persistence *silently* -- activities accumulate in memory and vanish at process
        exit. That is reachable in normal use: ``AgentClient`` computes its session
        directory but only wires it when it constructs the DataManager itself, so an
        injected DataManager needs this attachment to persist its activities.

        Also updates ``self._session_dir``, so the tracker rebuilds in ``clear()`` and
        ``clear_workspace()`` keep persisting rather than reverting to memory-only.

        Args:
            session_dir: Directory for provenance.jsonl / metadata.json. Created if absent.

        Returns:
            Path to the provenance file, or None when provenance is disabled entirely
            (``enable_provenance=False``), in which case there is nothing to attach.
        """
        self._session_dir = Path(session_dir)

        if self.provenance is None:
            logger.debug(
                "attach_session_dir: provenance is disabled on this DataManager; "
                "recorded the session dir for later but nothing will be written"
            )
            return None

        return self.provenance.attach_session_dir(self._session_dir)

    # Tool usage tracking via provenance
    #: What `agent` is recorded when a caller supplies none. Kept as the historical literal so
    #: that unwired call sites produce byte-identical records — see `log_tool_usage`.
    DEFAULT_PROVENANCE_AGENT = "data_manager"

    def log_tool_usage(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        description: str = None,
        ir: Optional["AnalysisStep"] = None,
        agent: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Log tool usage for reproducibility tracking via W3C-PROV provenance system.

        Args:
            tool_name: Name of the tool used
            parameters: Parameters used with the tool
            description: Optional description of what was done
            ir: Optional AnalysisStep Intermediate Representation for notebook export
            agent: Name of the agent that executed the tool (e.g.
                ``"transcriptomics_expert"``). Optional, and **defaults to the historical
                ``"data_manager"`` literal** — see the note below.

        Returns:
            Optional[Dict[str, Any]]: The created activity record, or None if no provenance

        Notes:
            The `ir` parameter enables automatic Jupyter notebook generation.
            Services should emit AnalysisStep objects alongside their results,
            and agents should pass these IR objects to this method for storage
            in the provenance record.

            **On `agent`.** This method previously hardcoded
            ``agent="data_manager"`` — the ``DataManagerV2`` object *writing* the record,
            not the agent that ran the tool, and unrelated to ``data_expert_agent``. Every
            activity in every session therefore carried the same string, so provenance could
            say *what* ran but never *who* ran it.

            ``create_activity()`` has always accepted a real ``agent`` parameter; this method
            simply had no way to forward one. It is **optional and defaults to the old
            literal** on purpose: ~116 call sites pass no agent, and making it required would
            break all of them at once. Tools are wired incrementally, and
            ``tests/unit/core/test_provenance_attribution_contract.py`` holds the audited set.
        """
        if self.provenance:
            activity_id = self.provenance.create_activity(
                activity_type=tool_name,
                agent=agent or self.DEFAULT_PROVENANCE_AGENT,
                parameters=parameters,
                description=description or f"{tool_name} operation",
                ir=ir,
            )

            # AQUADIF monitoring: observe provenance call (fail-open)
            if hasattr(self, "_aquadif_monitor") and self._aquadif_monitor is not None:
                try:
                    self._aquadif_monitor.record_provenance_call(
                        tool_name=tool_name,
                        has_real_ir=(ir is not None),
                    )
                except (
                    Exception
                ):  # nosec B110 # Advisory monitoring must not disrupt provenance recording.
                    pass  # Fail-open: monitor exception never disrupts provenance tracking

            # Find and return the activity dict
            for activity in self.provenance.activities:
                if activity["id"] == activity_id:
                    if ir is not None:
                        logger.info(
                            f"Tool usage logged with IR: {tool_name} (operation: {ir.operation})"
                        )
                    else:
                        logger.info(f"Tool usage logged: {tool_name}")
                    return activity

        return None

    def save_processed_data(
        self,
        processing_step: str,
        data_source: str = None,
        dataset_id: str = None,
        processing_params: Dict[str, Any] = None,
    ) -> str:
        """
        Save processed data with professional naming and enhanced metadata tracking.

        Args:
            processing_step: Current processing step (e.g., 'filtered', 'normalized')
            data_source: Data source (extracted from metadata if not provided)
            dataset_id: Dataset identifier (extracted from metadata if not provided)
            processing_params: Parameters used in the processing step

        Returns:
            str: Path to saved file
        """
        if not self.modalities:
            logger.warning("No modalities to save")
            return None

        try:
            import json

            from lobster.utils.file_naming import BioinformaticsFileNaming

            # Use the first modality if multiple exist
            modality_name = list(self.modalities.keys())[0]
            adata = self.modalities[modality_name]

            # Extract information from metadata or parameters
            data_source = data_source or "DATA"
            dataset_id = dataset_id or "unknown"

            # Generate professional filename with auto-selected extension
            filename = BioinformaticsFileNaming.generate_filename(
                data_source=data_source,
                dataset_id=dataset_id,
                processing_step=processing_step,
            )
            filepath = self.data_dir / filename

            # Save using appropriate backend
            self.save_modality(modality_name, filepath)

            # Create enhanced metadata
            enhanced_metadata = {
                "processing_step": processing_step,
                "data_source": data_source,
                "dataset_id": dataset_id,
                "saved_filename": filename,
                "saved_path": str(filepath),
                "save_timestamp": pd.Timestamp.now().isoformat(),
                "data_shape": list(adata.shape),
                "processing_params": processing_params or {},
                "processing_order": BioinformaticsFileNaming.get_processing_step_order(
                    processing_step
                ),
                "suggested_next_step": BioinformaticsFileNaming.suggest_next_step(
                    processing_step
                ),
                "modality_name": modality_name,
            }

            # Save metadata
            metadata_filename = BioinformaticsFileNaming.generate_metadata_filename(
                filename
            )
            metadata_path = self.data_dir / metadata_filename

            with open(metadata_path, "w") as f:
                json.dump(enhanced_metadata, f, indent=2, default=str)

            # Log the processing step
            with self._state_lock:
                self.processing_log.append(
                    f"Saved {processing_step} data: {adata.shape[0]} obs × {adata.shape[1]} vars -> {filename}"
                )

            logger.info(f"Processed data saved with professional naming: {filepath}")
            logger.info(
                f"Next suggested step: {enhanced_metadata['suggested_next_step']}"
            )

            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            return None

    @staticmethod
    def _package_recorded_pngs(store, ref, dest: Path, rel: str, entry: Dict) -> int:
        """Copy PNG renditions already recorded for ``ref`` (sha-verified); never render."""
        from lobster.core.figures.errors import safe_reason

        copied = 0
        for rec in store.list_renditions(ref):
            if (rec.get("settings") or {}).get("format") != "png":
                continue
            rendition_id = rec.get("rendition_id")
            try:
                data = store.read_rendition(ref, rendition_id)  # pending/changed raise
            except Exception as e:
                entry.setdefault("png_errors", []).append(
                    {"rendition_id": rendition_id, "reason": safe_reason(e)}
                )
                continue
            name = f"{rendition_id}.png"
            (dest / name).write_bytes(data)
            entry.setdefault("png_renditions", []).append(
                {
                    "rendition_id": rendition_id,
                    "file": f"{rel}/{name}",
                    "sha256": rec.get("sha256"),
                    "settings": rec.get("settings"),
                    "renderer": rec.get("renderer"),
                }
            )
            copied += 1
        return copied

    def _saved_figure_records(self) -> List[Dict[str, Any]]:
        """All saved revision records (metadata), restart-safe; empty without a plots dir.

        Delegates to the one native reader (``listing.read_saved_figures``): a plots-folder
        or catalog fault is a TYPED error, never an empty or partial list.
        """
        from lobster.core.figures.listing import read_saved_figures

        return list(read_saved_figures(self.workspace_path, lambda: self.figure_store))

    def _legacy_plot_names(self) -> Optional[List[str]]:
        """Flat (legacy) files in ``plots/``, read-only; ``None`` if they cannot be listed."""
        from lobster.core.figures.errors import safe_reason
        from lobster.core.figures.listing import legacy_plot_files

        try:
            return sorted(legacy_plot_files(self.workspace_path))
        except Exception as e:
            logger.warning("Could not list legacy plot files: %s", safe_reason(e))
            return None

    def list_workspace_files(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all files in the workspace organized by category.

        Categories: data, exports, cache and plots. ``plots`` lists managed figure
        revisions (``<figure_id>/<revision_id>/figure.json`` with ``figure_ref``) and
        read-only legacy flat files (``legacy: True``); the artifact manifest relies on
        it to report saved plots.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Files organized by category

        Raises:
            FigureError: the saved-figure inventory could not be read completely (typed,
                e.g. ``StorageError``); a partial inventory is never returned as complete.
        """
        files_by_category = {"data": [], "exports": [], "cache": [], "plots": []}

        # List data files
        for file_path in self.data_dir.iterdir():
            if file_path.is_file():
                files_by_category["data"].append(
                    {
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime,
                    }
                )

        # List export files
        for file_path in self.exports_dir.iterdir():
            if file_path.is_file():
                files_by_category["exports"].append(
                    {
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime,
                    }
                )

        # List cache files
        for file_path in self.cache_dir.iterdir():
            if file_path.is_file():
                files_by_category["cache"].append(
                    {
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime,
                    }
                )

        # Plots: managed revisions (from their records, restart-safe) + read-only legacy flat files.
        # Managed entries are staged from the ONE native reader, which returns every record or
        # raises a typed fault: a fault mid-scan is raised, never returned as a smaller,
        # apparently complete inventory (legacy files do not hide it).
        from lobster.core.figures.errors import FigureError, StorageError

        plots_dir = self.workspace_path / "plots"
        try:
            records = self._saved_figure_records()
        except FigureError:
            raise
        except Exception:  # an untyped store fault: typed here, its text never shown
            raise StorageError("saved figures could not be listed") from None
        managed = []
        for rec in records:
            rel = f"{rec['figure_id']}/{rec['revision_id']}/figure.json"
            file_path = plots_dir / rel
            try:
                stat = os.stat(file_path)
            except (FileNotFoundError, NotADirectoryError):
                stat = None  # listed from its record: recorded size, no mtime
            except OSError:
                raise StorageError("saved figure file unreadable") from None
            managed.append(
                {
                    "name": rel,
                    "path": str(file_path),
                    "size": (
                        stat.st_size if stat else rec.get("figure", {}).get("bytes", 0)
                    ),
                    "modified": stat.st_mtime if stat else None,
                    "figure_ref": f"{rec['figure_id']}@{rec['revision_id']}",
                    "label": rec.get("label"),
                    "kind": (rec.get("change") or {}).get("kind"),
                    "legacy": False,
                    # "pending" = save confirmation pending (not active yet)
                    "availability": rec.get("availability"),
                }
            )
        files_by_category["plots"].extend(managed)
        if plots_dir.is_dir():
            for file_path in plots_dir.iterdir():
                if file_path.is_file():
                    stat = file_path.stat()
                    files_by_category["plots"].append(
                        {
                            "name": file_path.name,
                            "path": str(file_path),
                            "size": stat.st_size,
                            "modified": stat.st_mtime,
                            "legacy": True,
                        }
                    )

        return files_by_category

    def _save_processing_log_and_provenance(self) -> Optional[bool]:
        """Return True if written, False if failed, None if nothing to persist."""
        if (
            self.processing_log
            or (self.provenance and self.provenance.activities)
            or self.modalities
            or self._modality_spilled
        ):
            # A data-bearing session may have empty history after a native spill.
            # Persist that real empty snapshot rather than claiming a nonexistent
            # processing log was saved.
            try:
                log_path = self.exports_dir / "processing_log.json"
                log_data = {
                    "processing_log": self.processing_log,
                    "provenance_summary": {
                        "n_activities": (
                            len(self.provenance.activities) if self.provenance else 0
                        ),
                        "activities": [
                            {
                                "type": act.get("type"),
                                "timestamp": act.get("timestamp"),
                                "parameters": act.get("parameters", {}),
                            }
                            for act in (
                                self.provenance.activities if self.provenance else []
                            )
                        ],
                    },
                    "timestamp": pd.Timestamp.now().isoformat(),
                }
                with open(log_path, "w") as f:
                    json.dump(log_data, f, indent=2, default=str)
                return True
            except Exception as e:
                logger.error(f"Failed to save processing log: {e}")
                return False
        return None

    def auto_save_state(
        self,
        progress_callback: Optional[callable] = None,
        force: bool = False,
    ) -> List[str]:
        """
        Automatically save current state including all modalities.

        Optimized to skip modalities that:
        - Were loaded from autosave files and haven't been modified
        - Already have up-to-date autosave files on disk

        Args:
            progress_callback: Optional callback(current, total, modality_name, status)
                for progress reporting. Status is one of: "saving", "skipped", "done", "error"
            force: If True, save all modalities regardless of dirty state

        Returns:
            List[str]: List of saved items (includes skipped items with indication)
        """
        saved_items = []
        skipped_items = []

        # When modality backend is attached, flush dirty through it
        if self._modality_backend is not None:
            for name in list(self._modality_dirty):
                if name in self.modalities:
                    try:
                        adata = self.modalities[name]
                        if getattr(adata, "isbacked", False):
                            adata = self.materialize_modality(name)
                        self._modality_backend.ingest_anndata(name, adata)
                        self._modality_dirty.discard(name)
                        saved_items.append(f"Modality '{name}': backend")
                    except Exception as e:
                        logger.error(f"Backend auto-save failed for '{name}': {e}")
                        saved_items.append(
                            f"{AUTOSAVE_NOT_SAVED_PREFIX}Modality '{name}': backend write failed"
                        )
            # Still save processing log/provenance below, but skip h5ad file writes
            log_outcome = self._save_processing_log_and_provenance()
            if log_outcome is True:
                saved_items.append("Processing log")
            elif log_outcome is False:
                saved_items.append(
                    f"{AUTOSAVE_NOT_SAVED_PREFIX}Processing log: write failed"
                )
            return saved_items

        # Before the first write: targets inside the managed figure folder are not
        # written (counted once in the result); one backup timestamp for this run.
        refused_resident, refused_spilled, backup_stamp = self._autosave_plan()
        save_kwargs = {} if backup_stamp is None else {"backup_stamp": backup_stamp}
        refused_count = 0
        # Targets outside the workspace are refused in the plan too, counted apart.
        outside = {
            name
            for name in (*refused_resident, *refused_spilled)
            if self._autosave_leads_out(name)
        }
        outside_count = 0

        modality_names = list(self.modalities.keys())
        total = len(modality_names)

        # Save all modalities (with smart skip logic)
        for idx, modality_name in enumerate(modality_names):
            try:
                # Determine the expected autosave path
                # If modality name already ends with _autosave, don't double-suffix
                if modality_name.endswith("_autosave"):
                    save_path = f"{modality_name}.h5ad"
                else:
                    save_path = f"{modality_name}_autosave.h5ad"

                full_save_path = self.data_dir / save_path

                # Check if we can skip this save
                should_skip = False
                skip_reason = ""

                if not force:
                    # Compute dirty + source FIRST so all cases can use them
                    source_file = self._modality_sources.get(modality_name)
                    is_dirty = modality_name in self._modality_dirty

                    # Case 0: Backed-mode modality — skip only if clean AND source exists
                    adata_obj = self.modalities.get(modality_name)
                    if (
                        getattr(adata_obj, "isbacked", False)
                        and not is_dirty
                        and source_file is not None
                        and source_file.exists()
                    ):
                        should_skip = True
                        skip_reason = "backed (on-disk, unmutated)"

                    # Case 1: Modality loaded from autosave file AND not dirty
                    is_from_autosave = source_file is not None and str(
                        source_file
                    ).endswith("_autosave.h5ad")

                    if is_from_autosave and not is_dirty:
                        # Check if source file still exists and matches target
                        if source_file.exists() and source_file == full_save_path:
                            should_skip = True
                            skip_reason = "unchanged autosave"
                        elif source_file.exists():
                            # Source exists but at different path - modality name might have
                            # _autosave suffix. Check if target already exists
                            if full_save_path.exists():
                                should_skip = True
                                skip_reason = "already saved"

                    # Case 2: Target autosave already exists and modality not dirty
                    elif full_save_path.exists() and not is_dirty:
                        # Check file modification time vs session start
                        file_mtime = full_save_path.stat().st_mtime
                        session_start = float(self.session_id)
                        if file_mtime >= session_start:
                            should_skip = True
                            skip_reason = "up-to-date"

                if should_skip:
                    skipped_items.append(f"Modality '{modality_name}': {skip_reason}")
                    if progress_callback:
                        progress_callback(idx + 1, total, modality_name, "skipped")
                    logger.debug(
                        f"Skipped auto-save for '{modality_name}': {skip_reason}"
                    )
                    continue

                if modality_name in refused_resident:  # stays dirty: not saved
                    if modality_name in outside:
                        outside_count += 1
                    else:
                        refused_count += 1
                    if progress_callback:
                        progress_callback(idx + 1, total, modality_name, "error")
                    continue

                # Report progress before saving
                if progress_callback:
                    progress_callback(idx + 1, total, modality_name, "saving")

                saved_path = self.save_modality(modality_name, save_path, **save_kwargs)
                saved_items.append(
                    f"Modality '{modality_name}': {Path(saved_path).name}"
                )

                # Clear dirty flag after successful save
                self._modality_dirty.discard(modality_name)

            except ManagedPathError:  # the target became managed after the plan
                logger.warning(_AUTOSAVE_REFUSED_LOG)
                refused_count += 1
                if progress_callback:
                    progress_callback(idx + 1, total, modality_name, "error")
            except Exception as e:
                logger.error(f"Failed to auto-save modality {modality_name}: {e}")
                saved_items.append(
                    f"{AUTOSAVE_NOT_SAVED_PREFIX}Modality '{modality_name}': write failed"
                )
                if progress_callback:
                    progress_callback(idx + 1, total, modality_name, "error")
            else:
                # A notification failure is not a write failure. Keep it out of
                # the save catch so completed writes cannot also be Not saved.
                if progress_callback:
                    progress_callback(idx + 1, total, modality_name, "done")

        # Spill any modalities beyond the configured memory budget.
        # are absent from self.modalities, so the loop above never sees them.
        # A spill file is already a complete h5ad, so copy it to the autosave
        # path rather than reloading the matrix into memory just to rewrite it.
        for modality_name, spill_path in list(self._modality_spilled.items()):
            try:
                if modality_name.endswith("_autosave"):
                    save_path = f"{modality_name}.h5ad"
                else:
                    save_path = f"{modality_name}_autosave.h5ad"
                full_save_path = self.data_dir / save_path
                source = Path(spill_path)
                if not source.exists():
                    logger.error(
                        f"Spill file for '{modality_name}' missing at {source}; "
                        f"cannot auto-save it"
                    )
                    saved_items.append(
                        f"{AUTOSAVE_NOT_SAVED_PREFIX}Modality '{modality_name}': spill file missing"
                    )
                    continue
                if full_save_path.exists() and full_save_path.samefile(source):
                    skipped_items.append(f"Modality '{modality_name}': already saved")
                    continue
                if modality_name in refused_spilled:  # stays spilled and dirty
                    if modality_name in outside:
                        outside_count += 1
                    else:
                        refused_count += 1
                    continue
                self._assert_spilled_autosave_allowed(full_save_path)
                full_save_path.parent.mkdir(parents=True, exist_ok=True)
                self._assert_save_allowed(full_save_path)
                _assert_not_a_folder(full_save_path)
                shutil.copy2(source, full_save_path)
                self._modality_dirty.discard(modality_name)
                saved_items.append(
                    f"Modality '{modality_name}': {full_save_path.name} (from spill)"
                )
            except ManagedPathError:  # the target became managed after the plan
                logger.warning(_AUTOSAVE_REFUSED_LOG)
                refused_count += 1
            except Exception as e:
                logger.error(
                    f"Failed to auto-save spilled modality '{modality_name}': {e}"
                )
                saved_items.append(
                    f"{AUTOSAVE_NOT_SAVED_PREFIX}Modality '{modality_name}': spill copy failed"
                )

        # Save processing log and provenance; never certify a swallowed failure.
        log_outcome = self._save_processing_log_and_provenance()
        if log_outcome is True:
            saved_items.append("Processing log")
        elif log_outcome is False:
            saved_items.append(
                f"{AUTOSAVE_NOT_SAVED_PREFIX}Processing log: write failed"
            )

        successes = [
            item
            for item in saved_items
            if not item.startswith(AUTOSAVE_NOT_SAVED_PREFIX)
        ]
        if successes:
            logger.info(f"Auto-saved: {', '.join(successes)}")
        failures = len(saved_items) - len(successes)
        if failures:
            logger.warning(f"Autosave incomplete: {failures} save operations failed")
        if skipped_items:
            logger.info(
                f"Skipped (already up-to-date): {len(skipped_items)} modalities"
            )
            logger.debug(f"Skipped details: {', '.join(skipped_items)}")

        # Return saved items plus skipped summary for CLI feedback
        all_results = saved_items.copy()
        if skipped_items:
            all_results.append(f"Skipped {len(skipped_items)} unchanged modalities")
        if refused_count:
            all_results.append(
                f"{AUTOSAVE_NOT_SAVED_PREFIX}{refused_count} modalities refused "
                "(managed figure folder)"
            )
        if outside_count:
            all_results.append(
                f"{AUTOSAVE_NOT_SAVED_PREFIX}{outside_count} modalities refused "
                "(outside the workspace)"
            )

        return all_results

    def _assert_spilled_autosave_allowed(self, full_save_path: Path) -> None:
        """The copy of a spill file to its auto-save target: its folder (created
        recursively), the file, and no folder standing at the file name."""
        _assert_dirs_allowed(self.workspace_path, full_save_path.parent)
        self._assert_save_allowed(full_save_path)
        _assert_not_a_folder(full_save_path)

    def _autosave_leads_out(self, name: str) -> bool:
        """True when the auto-save target of ``name`` is outside the workspace (a
        name registered directly, with '..' or an absolute spelling)."""
        return outside_workspace(
            self.workspace_path, self.data_dir / _autosave_file_name(name)
        )

    def _autosave_plan(self) -> tuple:
        """Classify every auto-save target before the first write (no effect).

        Returns the resident and the spilled modalities whose target, a folder created
        for it, or its planned MuData backup is inside the managed figure folder, and
        the backup timestamp of this run (read once; None unless the default backend
        is MuData). The skip rules still apply first when the run executes.
        """
        backend = self.backends.get(self.default_backend)
        backup_stamp = None
        if _is_mudata_backend(backend):
            backup_stamp = backend.backup_stamp()
        refused_resident, refused_spilled = set(), set()
        for name in self.modalities:
            save_path = _autosave_file_name(name)
            if self._autosave_leads_out(name):
                refused_resident.add(name)  # the name leads out of the workspace
                continue
            try:
                self._assert_save_allowed(save_path)
                if backend is None:
                    continue
                backend_path = self._backend_path(self.data_dir / save_path)
                self._assert_backend_destination_allowed(backend, backend_path)
                if backup_stamp is not None:
                    self._assert_planned_backup_allowed(
                        backend,
                        backend_path,
                        backup_stamp,
                        loading=(
                            name
                            if getattr(self.modalities[name], "isbacked", False)
                            else None
                        ),
                    )
            except ManagedPathError:
                refused_resident.add(name)
        for name in self._modality_spilled:
            if self._autosave_leads_out(name):
                refused_spilled.add(name)  # the name leads out of the workspace
                continue
            try:
                self._assert_spilled_autosave_allowed(
                    self.data_dir / _autosave_file_name(name)
                )
            except ManagedPathError:
                refused_spilled.add(name)
        return refused_resident, refused_spilled, backup_stamp

    def mark_modality_dirty(self, modality_name: str) -> None:
        """
        Mark a modality as modified, requiring save on next auto_save_state().

        This should be called by agents/services after modifying modality data.

        Args:
            modality_name: Name of the modality that was modified
        """
        with self._state_lock:
            if modality_name in self.modalities:
                self._modality_dirty.add(modality_name)
        logger.debug(f"Marked modality '{modality_name}' as dirty")

    def save_session(self) -> List[str]:
        """
        Save current session state.

        This is an alias for auto_save_state() to maintain API compatibility
        with UI components and legacy code.

        Returns:
            List[str]: List of saved items

        See Also:
            auto_save_state(): The underlying implementation method
        """
        return self.auto_save_state()

    def has_data(self) -> bool:
        """
        Check if any modalities are loaded.

        Returns:
            bool: True if modalities exist, False otherwise
        """
        return len(self.modalities) > 0

    def has_saveable_modalities(self) -> bool:
        """True when auto-save has a modality to write: resident, or spilled to disk
        to stay within the memory budget (``has_data()`` counts resident ones only)."""
        with self._state_lock:
            return bool(self.modalities) or bool(self._modality_spilled)

    # ========================================
    # PLOT MANAGEMENT - Use data_manager.plot_manager directly
    # ========================================
    # Plot management has been extracted to PlotManager (lobster/core/plot_manager.py).
    # Access via: data_manager.plot_manager.add_plot(), data_manager.plot_manager.save_plots_to_workspace(), etc.
    # Properties for backward compatibility: latest_plots, visualization_state

    # ========================================
    # CURRENT DATASET MANAGEMENT (Legacy Support)
    # ========================================

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of current dataset or all modalities.
        Handles various data types robustly (h5ad, sparse matrices, etc.).

        Returns:
            Dict[str, Any]: Summary information
        """
        if not self.has_data():
            return {"status": "No modalities loaded"}

        if self.current_dataset and self.current_dataset in self.modalities:
            # Summary for current dataset
            adata = self.modalities[self.current_dataset]

            try:
                metrics = self.get_quality_metrics(self.current_dataset)
            except Exception as e:
                logger.warning(
                    f"Could not get quality metrics for {self.current_dataset}: {e}"
                )
                metrics = {"error": str(e)}

            # Safely get memory usage for different matrix types
            memory_usage = self._get_safe_memory_usage(adata.X)

            # Safely get data type information
            data_type_info = self._get_safe_data_type_info(adata.X)

            summary = {
                "status": "Modality loaded",
                "modality_name": self.current_dataset,
                "shape": self._get_safe_shape(adata),
                "data_type": data_type_info,
                "sample_names": self._get_safe_obs_names(
                    adata
                ),  # CLI expects 'sample_names'
                "columns": self._get_safe_obs_columns(
                    adata
                ),  # CLI expects 'columns' for obs columns
                "memory_usage": memory_usage,
                "is_sparse": self._is_sparse_matrix(adata.X),
                "metadata_keys": (
                    list(self.current_metadata.keys()) if self.current_metadata else []
                ),
                "processing_log": (
                    self.processing_log[-5:] if self.processing_log else []
                ),
                # Additional fields for advanced usage
                "obs_names": self._get_safe_obs_names(adata),
                "var_names": self._get_safe_var_names(adata),
                "obs_columns": self._get_safe_obs_columns(adata),
                "var_columns": self._get_safe_var_columns(adata),
                "layers": self._get_safe_layers(adata),
                "obsm": self._get_safe_obsm(adata),
                "uns": self._get_safe_uns(adata),
                "quality_metrics": metrics,
            }
        else:
            # Summary for all modalities
            modality_summaries = {}
            for mod_name, adata in self.modalities.items():
                try:
                    modality_summaries[mod_name] = {
                        "shape": self._get_safe_shape(adata),
                        "data_type": self._get_safe_data_type_info(adata.X),
                        "obs_columns": (
                            len(adata.obs.columns)
                            if hasattr(adata, "obs") and adata.obs is not None
                            else 0
                        ),
                        "var_columns": (
                            len(adata.var.columns)
                            if hasattr(adata, "var") and adata.var is not None
                            else 0
                        ),
                        "layers": len(layer_names(adata)),
                        "memory_usage": self._get_safe_memory_usage(adata.X),
                        "is_sparse": self._is_sparse_matrix(adata.X),
                    }
                except Exception as e:
                    logger.warning(
                        f"Error getting summary for modality {mod_name}: {e}"
                    )
                    modality_summaries[mod_name] = {
                        "error": str(e),
                        "shape": (0, 0),
                        "data_type": "unknown",
                    }

            # Safely calculate totals
            total_obs, total_vars = 0, 0
            for adata in self.modalities.values():
                try:
                    shape = self._get_safe_shape(adata)
                    total_obs += shape[0]
                    total_vars += shape[1]
                except (
                    Exception
                ):  # nosec B112 # Best-effort shape summary ignores unavailable metadata.
                    continue

            summary = {
                "status": f"{len(self.modalities)} modalities loaded",
                "modalities": modality_summaries,
                "total_obs": total_obs,
                "total_vars": total_vars,
                "metadata_keys": (
                    list(self.current_metadata.keys()) if self.current_metadata else []
                ),
                "processing_log": (
                    self.processing_log[-5:] if self.processing_log else []
                ),
            }

        return summary

    def _get_safe_memory_usage(self, X) -> str:
        """Safely get memory usage for different matrix types."""
        try:
            if X is None:
                return "N/A (No data matrix)"

            # Handle sparse matrices
            if hasattr(X, "nnz") and hasattr(X, "data"):  # Likely a sparse matrix
                # For sparse matrices, calculate memory from data + indices + indptr
                if hasattr(X, "data") and hasattr(X.data, "nbytes"):
                    data_bytes = (
                        int(X.data.nbytes) if not isinstance(X.data.nbytes, Mock) else 0
                    )
                    if hasattr(X, "indices") and hasattr(X.indices, "nbytes"):
                        indices_bytes = (
                            int(X.indices.nbytes)
                            if not isinstance(X.indices.nbytes, Mock)
                            else 0
                        )
                        data_bytes += indices_bytes
                    if hasattr(X, "indptr") and hasattr(X.indptr, "nbytes"):
                        indptr_bytes = (
                            int(X.indptr.nbytes)
                            if not isinstance(X.indptr.nbytes, Mock)
                            else 0
                        )
                        data_bytes += indptr_bytes
                    return f"{data_bytes / 1024**2:.2f} MB (sparse)"
                else:
                    return f"~{X.nnz * 12 / 1024**2:.2f} MB (sparse, estimated)"

            # Handle dense matrices
            elif hasattr(X, "nbytes"):
                return f"{X.nbytes / 1024**2:.2f} MB (dense)"

            # Handle arrays with size and dtype
            elif hasattr(X, "size") and hasattr(X, "dtype"):
                bytes_estimate = X.size * X.dtype.itemsize
                return f"{bytes_estimate / 1024**2:.2f} MB (estimated)"

            # Fallback for other types
            else:
                return "Unknown (unsupported matrix type)"

        except Exception as e:
            return f"Error calculating memory: {str(e)}"

    def _get_safe_data_type_info(self, X) -> str:
        """Safely get data type information."""
        try:
            if X is None:
                return "None"

            # Check if it's a sparse matrix
            if hasattr(X, "nnz"):
                matrix_type = type(X).__name__
                if hasattr(X, "dtype"):
                    return f"{matrix_type}[{X.dtype}]"
                else:
                    return f"{matrix_type}[unknown dtype]"

            # Dense array/matrix
            elif hasattr(X, "dtype"):
                return f"{type(X).__name__}[{X.dtype}]"

            # Other types
            else:
                return str(type(X).__name__)

        except Exception:
            return "unknown"

    def _get_safe_shape(self, adata) -> tuple:
        """Safely get shape from AnnData object."""
        try:
            if hasattr(adata, "shape"):
                return adata.shape
            elif hasattr(adata, "n_obs") and hasattr(adata, "n_vars"):
                return (adata.n_obs, adata.n_vars)
            elif (
                hasattr(adata, "X")
                and adata.X is not None
                and hasattr(adata.X, "shape")
            ):
                return adata.X.shape
            else:
                return (0, 0)
        except Exception:
            return (0, 0)

    def _get_safe_obs_names(self, adata) -> list:
        """Safely get observation names."""
        try:
            if hasattr(adata, "obs_names") and adata.obs_names is not None:
                return list(adata.obs_names[:5])
            elif (
                hasattr(adata, "obs")
                and adata.obs is not None
                and hasattr(adata.obs, "index")
            ):
                return list(adata.obs.index[:5])
            else:
                return []
        except Exception:
            return []

    def _get_safe_var_names(self, adata) -> list:
        """Safely get variable names."""
        try:
            if hasattr(adata, "var_names") and adata.var_names is not None:
                return list(adata.var_names[:5])
            elif (
                hasattr(adata, "var")
                and adata.var is not None
                and hasattr(adata.var, "index")
            ):
                return list(adata.var.index[:5])
            else:
                return []
        except Exception:
            return []

    def _get_safe_obs_columns(self, adata) -> list:
        """Safely get observation columns."""
        try:
            if (
                hasattr(adata, "obs")
                and adata.obs is not None
                and hasattr(adata.obs, "columns")
            ):
                return list(adata.obs.columns[:10])
            else:
                return []
        except Exception:
            return []

    def _get_safe_var_columns(self, adata) -> list:
        """Safely get variable columns."""
        try:
            if (
                hasattr(adata, "var")
                and adata.var is not None
                and hasattr(adata.var, "columns")
            ):
                return list(adata.var.columns[:10])
            else:
                return []
        except Exception:
            return []

    def _get_safe_layers(self, adata) -> list:
        """Safely get layer names."""
        try:
            return layer_names(adata)
        except Exception:
            return []

    def _get_safe_obsm(self, adata) -> list:
        """Safely get obsm keys."""
        try:
            if hasattr(adata, "obsm") and adata.obsm is not None:
                return list(adata.obsm.keys())
            else:
                return []
        except Exception:
            return []

    def _get_safe_uns(self, adata) -> list:
        """Safely get uns keys."""
        try:
            if hasattr(adata, "uns") and adata.uns is not None:
                return list(adata.uns.keys())
            else:
                return []
        except Exception:
            return []

    def _is_sparse_matrix(self, X) -> bool:
        """Check if matrix is sparse."""
        try:
            return hasattr(X, "nnz") or "sparse" in str(type(X)).lower()
        except Exception:
            return False

    def set_current_dataset(self, modality_name: str) -> None:
        """
        Set the current active dataset/modality.

        Args:
            modality_name: Name of the modality to set as current

        Raises:
            ValueError: If modality not found
        """
        if modality_name not in self.modalities:
            raise ValueError(f"Modality '{modality_name}' not found")

        self.current_dataset = modality_name
        self.adata = self.modalities[modality_name]

        # Update session file to reflect current dataset access
        self._update_session_file("accessed")

        logger.info(f"Set current dataset to: {modality_name}")

    def set_data(
        self, data: pd.DataFrame, metadata: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """
        Legacy compatibility method - converts DataFrame to modality.

        Args:
            data: DataFrame containing biological data
            metadata: Optional metadata dictionary

        Returns:
            pd.DataFrame: The processed data that was set
        """
        try:
            if data is None or not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame.")

            if data.shape[0] == 0 or data.shape[1] == 0:
                raise ValueError("DataFrame is empty.")

            # Store metadata
            self.current_metadata = metadata or {}

            # Detect appropriate adapter based on data characteristics
            n_obs, n_vars = data.shape
            if n_vars > 5000:  # High gene count suggests single-cell
                adapter_name = "transcriptomics_single_cell"
                modality_name = "legacy_single_cell"
            elif n_vars < 1000:  # Low feature count might be proteomics
                adapter_name = "proteomics_ms"
                modality_name = "legacy_proteomics"
            else:  # Middle range suggests bulk RNA-seq
                adapter_name = "transcriptomics_bulk"
                modality_name = "legacy_bulk"

            # Load as modality
            self.load_modality(
                name=modality_name,
                source=data,
                adapter=adapter_name,
                validate=True,
                **self.current_metadata,
            )

            # Set as current dataset
            self.set_current_dataset(modality_name)

            # Log the processing step
            with self._state_lock:
                self.processing_log.append(
                    f"Legacy data loaded: {data.shape[0]} samples × {data.shape[1]} features"
                )

            return data

        except Exception as e:
            logger.exception(f"Error in set_data: {e}")
            raise

    # ========================================
    # METADATA AND SESSION MANAGEMENT
    # ========================================

    def store_metadata(
        self,
        dataset_id: str,
        metadata: Dict[str, Any],
        validation_info: Dict[str, Any] = None,
    ) -> None:
        """
        Store metadata for a dataset in the metadata store.

        Args:
            dataset_id: Unique identifier for the dataset
            metadata: Metadata dictionary
            validation_info: Optional validation information
        """
        entry = {
            "metadata": metadata,
            "validation_result": validation_info or {},
            "fetch_timestamp": datetime.now().isoformat(),
            "stored_by": "DataManagerV2",
        }
        with self._state_lock:
            self.metadata_store[dataset_id] = entry
        logger.info(f"Stored metadata for dataset: {dataset_id}")

    def _store_geo_metadata(
        self, geo_id: str, metadata: Dict[str, Any], stored_by: str, **kwargs
    ) -> MetadataEntry:
        """
        Store GEO metadata with enforced consistent structure.

        This helper method ensures all GEO metadata is stored with the same
        nested structure, preventing KeyError bugs when retrieving metadata.

        Args:
            geo_id: GEO dataset identifier (e.g., 'GSE12345')
            metadata: Raw GEO metadata dictionary from GEOparse (must not be None)
            stored_by: Component storing the metadata (for tracking)
            **kwargs: Optional additional fields:
                - validation_result (Dict): Validation information
                - strategy_config (Dict): Download strategy configuration
                - modality_detection (Dict): Modality detection results
                - concatenation_decision (Dict): Concatenation strategy

        Returns:
            MetadataEntry: The stored metadata entry

        Raises:
            ValueError: If metadata is None

        Example:
            >>> entry = data_manager._store_geo_metadata(
            ...     geo_id="GSE12345",
            ...     metadata=geoparse_metadata,
            ...     stored_by="_check_platform_compatibility",
            ...     modality_detection={"modality": "single_cell", "confidence": 0.95}
            ... )
        """
        if metadata is None:
            raise ValueError(
                f"metadata dict is required for _store_geo_metadata (geo_id={geo_id})"
            )

        # Warn if validation_result is explicitly passed as None
        if "validation_result" in kwargs and kwargs["validation_result"] is None:
            logger.warning(f"validation_result is None for {geo_id}")

        # Create entry with required fields
        entry: MetadataEntry = {
            "metadata": metadata.copy(),
            "fetch_timestamp": kwargs.get(
                "fetch_timestamp", datetime.now().isoformat()
            ),
            "stored_by": stored_by,
        }

        # Add optional fields if provided
        if "validation_result" in kwargs:
            entry["validation_result"] = kwargs["validation_result"]

        if "strategy_config" in kwargs:
            entry["strategy_config"] = kwargs["strategy_config"]

        if "modality_detection" in kwargs:
            entry["modality_detection"] = kwargs["modality_detection"]

        if "concatenation_decision" in kwargs:
            entry["concatenation_decision"] = kwargs["concatenation_decision"]

        # Store with nested structure
        with self._state_lock:
            self.metadata_store[geo_id] = entry

        logger.debug(
            f"Stored GEO metadata for {geo_id} with nested structure "
            f"(stored_by: {stored_by}, keys: {list(entry.keys())})"
        )

        return entry

    def _enrich_geo_metadata(self, geo_id: str, **fields) -> Optional[MetadataEntry]:
        """
        Add or update fields on an existing metadata entry.

        Use this instead of direct metadata_store assignments to maintain
        centralized write control and consistent structure.

        Args:
            geo_id: GEO dataset identifier
            **fields: Key-value pairs to add/update on the existing entry

        Returns:
            Optional[MetadataEntry]: Updated entry, or None if entry does not exist
        """
        entry = self._get_geo_metadata(geo_id)
        if entry is None:
            logger.warning(f"Cannot enrich metadata for {geo_id}: no existing entry")
            return None
        entry.update(fields)
        with self._state_lock:
            self.metadata_store[geo_id] = entry  # Single controlled re-assignment point
        logger.debug(
            f"Enriched GEO metadata for {geo_id} with fields: {list(fields.keys())}"
        )
        return entry

    def _get_geo_metadata(self, geo_id: str) -> Optional[MetadataEntry]:
        """
        Safely retrieve GEO metadata with structure validation.

        This helper method retrieves GEO metadata from the store and validates
        that it has the expected nested structure.

        Args:
            geo_id: GEO dataset identifier

        Returns:
            Optional[MetadataEntry]: Metadata entry if found and valid, None otherwise

        Example:
            >>> entry = data_manager._get_geo_metadata("GSE12345")
            >>> if entry:
            ...     raw_metadata = entry["metadata"]
            ...     strategy = entry.get("strategy_config", {})
        """
        if geo_id not in self.metadata_store:
            logger.debug(f"No metadata found for {geo_id}")
            return None

        stored_entry = self.metadata_store[geo_id]

        # Validate structure
        if not isinstance(stored_entry, dict):
            logger.warning(
                f"Metadata for {geo_id} is not a dictionary: {type(stored_entry)}"
            )
            return None

        if "metadata" not in stored_entry:
            logger.warning(
                f"Metadata for {geo_id} missing 'metadata' key. "
                f"Found keys: {list(stored_entry.keys())}. "
                f"This indicates a structure mismatch bug."
            )
            return None

        return stored_entry

    def get_stored_metadata(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve stored metadata for a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Optional[Dict[str, Any]]: Metadata if found, None otherwise
        """
        return self.metadata_store.get(dataset_id)

    def list_stored_datasets(self) -> List[str]:
        """
        List all datasets with stored metadata.

        Returns:
            List[str]: List of dataset IDs with metadata
        """
        return list(self.metadata_store.keys())

    # ========================================
    # MACHINE LEARNING - Use MLPreparationService directly
    # ========================================
    # ML methods have been extracted to MLPreparationService (lobster/services/ml/ml_preparation_service.py).
    # Instantiate service and call methods directly:
    #   from lobster.services.ml.ml_preparation_service import MLPreparationService
    #   service = MLPreparationService()
    #   result, stats, ir = service.check_ml_readiness(adata, modality_name)

    def get_technical_summary(self) -> str:
        """
        Generate a technical summary of all analysis and processing.

        Returns:
            str: Formatted technical summary
        """
        summary = "# DataManagerV2 Technical Summary\n\n"

        # Add modality information
        if self.modalities:
            summary += "## Loaded Modalities\n\n"
            for name, adata in self.modalities.items():
                summary += f"### {name}\n"
                summary += f"- Shape: {adata.n_obs} obs × {adata.n_vars} vars\n"
                summary += (
                    f"- Memory usage: {adata.X.nbytes / 1024**2:.2f} MB\n"
                    if hasattr(adata.X, "nbytes")
                    else ""
                )
                if adata.obs.columns.tolist():
                    summary += f"- Observation metadata: {', '.join(list(adata.obs.columns)[:5])}\n"
                if adata.var.columns.tolist():
                    summary += f"- Variable metadata: {', '.join(list(adata.var.columns)[:5])}\n"
                layers = layer_names(adata)
                if layers:
                    summary += f"- Data layers: {', '.join(layers)}\n"
                summary += "\n"

        # Add processing log
        if self.processing_log:
            summary += "## Processing Log\n\n"
            for entry in self.processing_log:
                summary += f"- {entry}\n"
            summary += "\n"

        # Add provenance information with detailed activities
        if self.provenance and self.provenance.activities:
            summary += "## Provenance-Tracked Activities\n\n"
            summary += f"**Summary**: {len(self.provenance.activities)} activities, "
            summary += f"{len(self.provenance.entities)} entities, "
            summary += f"{len(self.provenance.agents)} agents\n\n"

            for i, activity in enumerate(self.provenance.activities, 1):
                activity_type = activity.get("type", "unknown")
                timestamp = activity.get("timestamp", "N/A")
                summary += f"### {i}. {activity_type} ({timestamp})\n\n"

                if activity.get("description"):
                    summary += f"{activity['description']}\n\n"

                if activity.get("parameters"):
                    summary += "**Parameters:**\n\n"
                    for param_name, param_value in activity["parameters"].items():
                        # Format parameter value based on its type
                        if (
                            isinstance(param_value, (list, tuple))
                            and len(param_value) > 5
                        ):
                            param_str = f"[{', '.join(str(x) for x in param_value[:5])}...] (length: {len(param_value)})"
                        else:
                            param_str = str(param_value)
                        summary += f"- {param_name}: {param_str}\n"
                    summary += "\n"

        return summary

    def create_data_package(
        self,
        output_dir: str = None,
        progress_callback=None,
        include_png: bool = True,
        compression: str = "gzip",
        include_provenance: bool = True,
        reuse_autosave: bool = True,
    ) -> str:
        """
        Create a comprehensive data package with modalities, plots, and analysis summary.

        Args:
            output_dir: Directory to save the package (defaults to exports_dir)
            progress_callback: Optional callback function to report progress (receives message string)
            include_png: Kept for compatibility; ``False`` (``--no-png``) disables PNG
                generation only. No new PNG is rendered locally anyway (no Chrome/Kaleido);
                the index records ``png: not_rendered_locally``. PNG renditions already
                recorded for a revision are ALWAYS included with their exact, sha-verified
                bytes; anything not copied is listed in the README and in
                ``self.last_package_report``.
            compression: H5AD compression method ('gzip', 'lzf', None) (default: 'gzip')
            include_provenance: Whether to include provenance information (default: True)
            reuse_autosave: Whether to copy existing autosave files instead of re-serializing (default: True)

        Returns:
            str: Path to the created zip file

        Raises:
            ManagedPathError: The package, its folder, its staging folder or a payload
                file is inside the managed figure folder (or a modality name would
                place its payload file outside the package)
        """
        legacy_names = self._legacy_plot_names()
        if not self.has_data() and not self._saved_figure_records():
            if legacy_names:
                raise ValueError(
                    "No data or saved figures to export; "
                    f"{len(legacy_names)} legacy plot file(s) in plots/ are not "
                    "supported by export (left unchanged)"
                )
            if legacy_names is None:
                raise ValueError(
                    "No data or saved figures to export; legacy plot files in plots/ "
                    "could not be listed"
                )
            raise ValueError("No data or plots to export")

        if progress_callback:
            progress_callback("Preparing export package...")

        # Create a timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_dir = self.exports_dir if output_dir is None else Path(output_dir)
        zip_filename = package_dir / f"lobster_analysis_package_{timestamp}.zip"

        # Before anything is created: every payload name stays a plain file name, and
        # the package, its folder and the staging folder are outside plots/<figure>/.
        for name in self.modalities:
            if not all(map(_is_plain_file_name, _package_payload_names(name))):
                raise _managed_refusal() from None
        staging = _package_staging_parent()
        _assert_dirs_allowed(self.workspace_path, staging)
        _assert_dirs_allowed(self.workspace_path, package_dir)
        self._assert_save_allowed(Path.cwd() / zip_filename)

        # Use exports directory if not specified
        owned_dir = None  # the folder this call created (removed if left empty)
        if output_dir is None:
            output_dir = str(self.exports_dir)
        else:
            os.makedirs(package_dir.parent, exist_ok=True)
            try:
                os.mkdir(package_dir)
                owned_dir = package_dir
            except FileExistsError:
                if not package_dir.is_dir():
                    raise

        try:
            return self._write_data_package(
                staging,
                zip_filename,
                legacy_names,
                progress_callback,
                include_png,
                compression,
                include_provenance,
                reuse_autosave,
            )
        except ManagedPathError:
            if owned_dir is not None:
                try:
                    os.rmdir(owned_dir)  # only when still empty
                except OSError:
                    pass
            raise

    def _assert_package_file_allowed(self, path: Path) -> None:
        """Right before a payload file is written (or copied) into the package."""
        self._assert_save_allowed(path)
        _assert_not_a_folder(path)

    def _write_data_package(
        self,
        staging: str,
        zip_filename: Path,
        legacy_names: Optional[List[str]],
        progress_callback,
        include_png: bool,
        compression: str,
        include_provenance: bool,
        reuse_autosave: bool,
    ) -> str:
        """Build the package of :meth:`create_data_package` in a new staging folder."""
        # Create a temporary directory for files
        with tempfile.TemporaryDirectory(dir=staging) as temp_dir:
            temp_path = Path(temp_dir)
            for name in self.modalities:  # every payload file, before the first write
                for leaf in _package_payload_names(name):
                    self._assert_package_file_allowed(temp_path / "modalities" / leaf)

            if progress_callback:
                progress_callback("Creating technical summary...")

            # Save technical summary
            with open(temp_path / "technical_summary.md", "w") as f:
                f.write(self.get_technical_summary())

            # Save all modalities
            if self.modalities:
                if progress_callback:
                    progress_callback(
                        f"Exporting {len(self.modalities)} data modalities..."
                    )

                modalities_dir = temp_path / "modalities"
                modalities_dir.mkdir()

                # Get H5AD backend for proper sanitization
                h5ad_backend = self.backends.get("h5ad")

                for i, (name, adata) in enumerate(self.modalities.items(), 1):
                    if progress_callback:
                        progress_callback(
                            f"Saving modality {i}/{len(self.modalities)}: {name}"
                        )
                    try:
                        h5ad_path = modalities_dir / f"{name}.h5ad"
                        reused = False

                        # Optimization: Try to reuse existing autosave file if available
                        if reuse_autosave:
                            # Check for existing autosave file
                            source_file = self._modality_sources.get(name)
                            is_dirty = name in self._modality_dirty

                            # Check autosave naming patterns
                            autosave_candidates = [
                                self.data_dir / f"{name}_autosave.h5ad",
                                self.data_dir / f"{name}.h5ad",
                            ]
                            if source_file:
                                autosave_candidates.insert(0, source_file)

                            for autosave_path in autosave_candidates:
                                if (
                                    autosave_path.exists()
                                    and not is_dirty
                                    and autosave_path != h5ad_path
                                ):
                                    # Copy existing file instead of re-serializing
                                    import shutil

                                    self._assert_package_file_allowed(h5ad_path)
                                    shutil.copy2(autosave_path, h5ad_path)
                                    reused = True
                                    logger.debug(
                                        f"Reused existing file for '{name}': {autosave_path.name}"
                                    )
                                    break

                        # If not reused, save with proper H5AD backend sanitization
                        if not reused:
                            self._assert_package_file_allowed(h5ad_path)
                            if h5ad_backend:
                                # Use backend's save method for proper sanitization
                                # (handles ArrowExtensionArray, None columns, mixed types, etc.)
                                h5ad_backend.save(
                                    adata, h5ad_path, compression=compression
                                )
                            else:
                                # Fallback: direct save (should not happen in normal operation)
                                logger.warning(
                                    f"H5AD backend not available, using direct save for '{name}'"
                                )
                                # Convert ArrowExtensionArray before direct write
                                try:
                                    from lobster.core.utils.h5ad_utils import (
                                        convert_arrow_to_standard,
                                    )

                                    adata = convert_arrow_to_standard(adata)
                                except (
                                    Exception
                                ):  # nosec B110 # Optional Arrow compatibility conversion may fall back to the original data.
                                    pass
                                adata.write_h5ad(h5ad_path, compression=compression)

                        # Save metadata (optimized to avoid unnecessary list conversions)
                        metadata_path = modalities_dir / f"{name}_metadata.json"
                        modality_metadata = {
                            "shape": adata.shape,
                            "n_obs": adata.n_obs,
                            "n_vars": adata.n_vars,
                            "obs_columns": adata.obs.columns.tolist(),
                            "var_columns": adata.var.columns.tolist(),
                            "layers": layer_names(adata),
                            "obsm_keys": list(adata.obsm.keys()) if adata.obsm else [],
                            "uns_keys": list(adata.uns.keys()) if adata.uns else [],
                            "is_sparse": hasattr(adata.X, "toarray"),
                            "data_type": (
                                str(adata.X.dtype)
                                if hasattr(adata.X, "dtype")
                                else "unknown"
                            ),
                            "reused_from_autosave": reused,
                        }
                        self._assert_package_file_allowed(metadata_path)
                        with open(metadata_path, "w") as f:
                            json.dump(modality_metadata, f, indent=2, default=str)

                    except ManagedPathError:
                        raise  # the whole package is refused (fixed reason)
                    except Exception as e:
                        logger.error(f"Failed to save modality {name}: {e}")
                        if progress_callback:
                            progress_callback(f"Error saving {name}: {str(e)[:50]}...")

            # Save plots: every saved revision with its EXACT committed bytes (sha-verified on
            # read) plus an HTML view written once on this explicit export path. A failed view
            # never drops the revision; it is reported separately. No new PNG is rendered
            # locally; PNG renditions already recorded are kept (exact, sha-verified bytes).
            self.last_package_report = None  # set once this package is fully described
            png_included = 0
            plots_index: List[Dict[str, Any]] = []
            revisions = self._saved_figure_records()
            if revisions:
                from lobster.core.figures.errors import safe_reason
                from lobster.core.figures.renditions import ensure_html_rendition

                if progress_callback:
                    progress_callback(
                        f"Exporting {len(revisions)} saved figure revision(s)..."
                    )
                plots_dir = temp_path / "plots"
                plots_dir.mkdir()
                store = self.figure_store
                for rec in revisions:
                    fid, rid = rec["figure_id"], rec["revision_id"]
                    ref = f"{fid}@{rid}"
                    entry = {
                        "figure_ref": ref,
                        "label": rec.get("label"),
                        "kind": (rec.get("change") or {}).get("kind"),
                        "parents": rec.get("parents", []),
                        "created_at": rec.get("created_at"),
                        "files": {},
                        "html": None,
                        "png": "not_rendered_locally" if include_png else None,
                        "availability": rec.get("availability"),
                    }
                    try:
                        rev_bytes, fig_bytes, cfg_bytes = store.read(ref)
                        dest = plots_dir / fid / rid
                        dest.mkdir(parents=True)
                        for name, data in (
                            ("revision.json", rev_bytes),
                            ("figure.json", fig_bytes),
                            ("config.json", cfg_bytes),
                        ):
                            (dest / name).write_bytes(data)
                            entry["files"][name] = f"{fid}/{rid}/{name}"
                        view = ensure_html_rendition(store, ref)
                        if view.status == "available":
                            (dest / "figure.html").write_bytes(
                                store.read_rendition(ref, view.rendition_id)
                            )
                            entry["html"] = f"{fid}/{rid}/figure.html"
                        else:
                            entry["html_error"] = view.reason
                        png_included += self._package_recorded_pngs(
                            store, ref, dest, f"{fid}/{rid}", entry
                        )
                        entry["not_included"] = _lifecycle_refs_not_included(
                            store, json.loads(rev_bytes)
                        )
                    except Exception as e:
                        entry["error"] = safe_reason(e)
                        logger.warning(
                            "Figure %s not exported: %s", ref, entry["error"]
                        )
                    plots_index.append(entry)

                with open(plots_dir / "index.json", "w") as f:
                    json.dump(plots_index, f, indent=2)
                with open(plots_dir / "README.md", "w") as f:
                    f.write("# Saved figures\n\n")
                    f.write(_VIEW_PACKAGE_SCOPE + "\n\n")
                    for idx, info in enumerate(plots_index, 1):
                        f.write(f"## {idx}. {info['label'] or info['figure_ref']}\n\n")
                        f.write(f"- Ref: {info['figure_ref']} ({info['kind']})\n")
                        f.write(f"- Created: {info['created_at']}\n")
                        if info.get("html"):
                            f.write(f"- View: [{info['html']}]({info['html']})\n")
                        elif info.get("html_error"):
                            f.write(f"- View: not written ({info['html_error']})\n")
                        for png in info.get("png_renditions", []):
                            f.write(
                                f"- PNG (recorded): [{png['file']}]({png['file']})\n"
                            )
                        for png in info.get("png_errors", []):
                            f.write(
                                f"- PNG (recorded) NOT copied: {png['rendition_id']} "
                                f"({png['reason']})\n"
                            )
                        if info.get("not_included"):
                            f.write(
                                "- Not included (referenced, not copied): "
                                f"{_describe_refs(info['not_included'])}\n"
                            )
                        if info.get("error"):
                            f.write(f"- Not exported: {info['error']}\n")
                        f.write("\n")

            if progress_callback:
                progress_callback("Saving workspace metadata...")

            # Save workspace status
            with open(temp_path / "workspace_status.json", "w") as f:
                json.dump(self.get_workspace_status(), f, indent=2, default=str)

            # Save provenance if available and requested
            if include_provenance and self.provenance:
                with open(temp_path / "provenance.json", "w") as f:
                    json.dump(self.provenance.to_dict(), f, indent=2, default=str)

            png_not_copied = sum(len(i.get("png_errors", [])) for i in plots_index)
            not_copied = _not_copied_section(
                plots_index, legacy_names
            ) + _not_included_section(plots_index)
            self.last_package_report = {
                "view_only": True,
                "lifecycle_refs_not_included": sum(
                    len(i.get("not_included") or []) for i in plots_index
                ),
                "figure_revisions": len(plots_index),
                "figures_not_exported": sum(1 for i in plots_index if i.get("error")),
                "html_views_not_written": sum(
                    1 for i in plots_index if i.get("html_error")
                ),
                "recorded_png_included": png_included,
                "recorded_png_not_copied": png_not_copied,
                # None = the legacy files could not be listed (never reported as 0)
                "legacy_plot_files_not_copied": (
                    None if legacy_names is None else len(legacy_names)
                ),
            }

            # Create professional README for the export package
            readme_content = f"""# Lobster Analysis Export Package

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Contents

### Data Files (H5AD Format)
- **Format**: H5AD (HDF5-based AnnData format)
- **Compression**: {compression or "none"}
- **Recommendation**: Open with scanpy, cellxgene, or other single-cell analysis tools

### Visualizations
- **Saved figures**: `plots/<figure>/<revision>/` holds each saved revision's exact `revision.json`, `figure.json` and `config.json`; `plots/index.json` lists them
- **HTML Files**: an interactive Plotly view per revision (`figure.html`), written during this export; a revision whose view could not be written is listed with `html_error`
- **PNG Files**: {_png_readme_line(include_png, png_included, png_not_copied)}

{not_copied}
### Metadata
- **technical_summary.md**: Analysis overview and methods
- **workspace_status.json**: System configuration and data provenance
{"- **provenance.json**: Analysis history recorded by Lobster" if include_provenance else ""}

## Usage

### Loading Data in Python
```python
import scanpy as sc
adata = sc.read_h5ad('modalities/your_dataset.h5ad')
```

### Loading Data in R
```r
library(anndata)
adata <- read_h5ad('modalities/your_dataset.h5ad')
```

### Viewing Plots
Open HTML files in any web browser for interactive exploration.

## Citation
Generated with Lobster AI - Multi-Agent Bioinformatics Analysis Platform
https://github.com/OmicsOS/lobster
"""

            with open(temp_path / "README.md", "w") as f:
                f.write(readme_content)

            if progress_callback:
                progress_callback("Creating final package...")

            # Create the zip file
            self._assert_save_allowed(Path.cwd() / zip_filename)
            with zipfile.ZipFile(zip_filename, "w") as zipf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(temp_path)
                        zipf.write(file_path, arcname)

        logger.info(f"Data package created: {zip_filename}")
        return str(zip_filename)

    # Formats that are scanned/loaded as tabular data (not h5ad, not specialized binary)
    _TABULAR_FORMATS = frozenset({"csv", "tsv", "parquet", "xlsx"})

    def _get_scannable_extensions(self) -> set:
        """Derive scannable file extensions from registered adapters."""
        extensions = set()
        for adapter in self.adapters.values():
            for fmt in adapter.get_supported_formats():
                extensions.add(fmt)
        # Always include these common formats even if no adapter explicitly declares them
        extensions.update({"h5ad", "csv", "tsv", "parquet"})
        return extensions

    def _extract_tabular_metadata(self, file_path: Path) -> dict:
        """Extract lightweight metadata from a tabular file (CSV/TSV/parquet/Excel).

        Reads only headers and file stats — does not load the full dataset into memory.
        For CSV/TSV files >50 MB, shape extraction is skipped to avoid unbounded I/O.
        """
        stat = file_path.stat()
        ext = file_path.suffix.lower().lstrip(".")
        file_size_mb = stat.st_size / 1e6
        base_meta = {
            "path": str(file_path),
            "size_mb": file_size_mb,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "type": ext,
        }

        # Try to extract shape cheaply per format
        shape = (0, 0)
        try:
            if ext == "parquet":
                import pyarrow.parquet as pq

                meta = pq.read_metadata(file_path)
                schema = pq.read_schema(file_path)
                shape = (meta.num_rows, len(schema))
            elif ext in ("csv", "tsv"):
                # Skip full-file line counting for large files (>50 MB)
                if file_size_mb > 50:
                    # Header-only: get column count, estimate rows from file size
                    with open(file_path, "r") as f:
                        header = f.readline()
                    sep = "\t" if ext == "tsv" or "\t" in header else ","
                    n_cols = len(header.split(sep))
                    # Rough estimate: avg line ~100 bytes
                    estimated_rows = max(0, int(stat.st_size / 100) - 1)
                    shape = (estimated_rows, max(0, n_cols - 1))
                else:
                    with open(file_path, "r") as f:
                        header = f.readline()
                        n_lines = sum(1 for _ in f)  # count remaining lines
                    sep = "\t" if ext == "tsv" or "\t" in header else ","
                    n_cols = len(header.split(sep))
                    shape = (max(0, n_lines), max(0, n_cols - 1))
        except ImportError as e:
            logger.debug(f"Missing dependency for {file_path.name} metadata: {e}")
        except Exception as e:
            logger.debug(f"Could not extract shape from {file_path.name}: {e}")

        base_meta["shape"] = shape
        return base_meta

    def _scan_workspace(self) -> Dict[str, Dict]:
        """
        Scan workspace for available datasets without loading them.

        Discovers H5AD files (via h5py for efficient metadata reading) and tabular
        files (CSV, TSV, parquet, XLSX) via lightweight header inspection.

        Returns:
            Dict mapping dataset names to their metadata
        """
        with self._measure_step("dm:scan_workspace"):
            datasets = {}
            data_dir = self.workspace_path / "data"

            if not data_dir.exists():
                return datasets

            corrupted_autosaves_cleaned = 0

            # --- H5AD files: existing h5py-based metadata extraction (unchanged) ---
            for h5ad_file in data_dir.glob("*.h5ad"):
                try:
                    is_autosave = h5ad_file.name.endswith("_autosave.h5ad")
                    import h5py

                    with h5py.File(h5ad_file, "r") as f:
                        if "X" in f:
                            if isinstance(f["X"], h5py.Dataset):
                                shape = f["X"].shape
                            elif isinstance(f["X"], h5py.Group):
                                shape = (
                                    tuple(f["X"].attrs["shape"])
                                    if "shape" in f["X"].attrs
                                    else (0, 0)
                                )
                            else:
                                shape = (0, 0)
                        else:
                            shape = (0, 0)

                        stat = h5ad_file.stat()
                        if not is_autosave:
                            datasets[h5ad_file.stem] = {
                                "path": str(h5ad_file),
                                "size_mb": stat.st_size / 1e6,
                                "shape": shape,
                                "modified": datetime.fromtimestamp(
                                    stat.st_mtime
                                ).isoformat(),
                                "type": "h5ad",
                            }
                except HdfReadRefusal:
                    raise
                except Exception as e:
                    is_autosave = h5ad_file.name.endswith("_autosave.h5ad")
                    if is_autosave:
                        try:
                            h5ad_file.unlink()
                            corrupted_autosaves_cleaned += 1
                            logger.debug(
                                f"Removed corrupted autosave during scan: {h5ad_file.name}"
                            )
                        except OSError:
                            pass
                    else:
                        logger.debug(f"Could not scan {h5ad_file.name}: {e}")

            if corrupted_autosaves_cleaned > 0:
                logger.debug(
                    f"Scan cleaned up {corrupted_autosaves_cleaned} corrupted autosave file(s)"
                )

            # --- Non-H5AD tabular files ---
            scannable = self._get_scannable_extensions()
            tabular_to_scan = self._TABULAR_FORMATS & scannable

            for ext in sorted(tabular_to_scan):
                for data_file in data_dir.glob(f"*.{ext}"):
                    stem = data_file.stem
                    if stem in datasets:
                        continue  # h5ad takes precedence
                    try:
                        datasets[stem] = self._extract_tabular_metadata(data_file)
                    except Exception as e:
                        logger.debug(f"Could not scan {data_file.name}: {e}")

            # Update self.available_datasets for backward compatibility
            self.available_datasets = datasets
            return datasets

    def get_available_datasets(self, force_refresh: bool = False) -> Dict[str, Dict]:
        """Get available datasets with TTL-based caching.

        Cached scans avoid repeating filesystem work until the cache expires.

        Args:
            force_refresh: If True, bypass cache and rescan filesystem

        Returns:
            Dict mapping dataset names to their metadata
        """
        with self._measure_step("dm:get_available_datasets"):
            current_time = time.time()
            cache_age = current_time - self._scan_timestamp

            # Check if cache is valid (not expired and not force refresh)
            cache_valid = (
                not force_refresh
                and self._available_datasets_cache is not None
                and cache_age < self._scan_ttl
            )

            if cache_valid:
                logger.debug(
                    f"Workspace cache hit (age: {cache_age:.1f}s, TTL: {self._scan_ttl}s)"
                )
                return self._available_datasets_cache

            # Cache miss or expired - perform scan
            logger.debug(
                f"Workspace cache {'forced refresh' if force_refresh else 'miss'}  "
                f"(age: {cache_age:.1f}s, TTL: {self._scan_ttl}s)"
            )
            self._available_datasets_cache = self._scan_workspace()
            self._scan_timestamp = current_time

            return self._available_datasets_cache

    def invalidate_scan_cache(self) -> None:
        """Force refresh on next workspace scan access.

        Call after operations that modify workspace contents, such as loading new data or
        deleting datasets, to keep cached results consistent with the filesystem.

        Usage:
            - After saving new datasets
            - After deleting datasets
            - When user explicitly requests refresh (e.g., /workspace list --refresh)
        """
        self._available_datasets_cache = None
        self._scan_timestamp = 0
        logger.debug(
            "Workspace scan cache invalidated - next access will trigger fresh scan"
        )

    def _load_session_metadata(self) -> None:
        """Load session metadata from file (multi-process safe)."""
        try:
            with queue_file_lock(self._session_lock, self._session_lock_path):
                with open(self.session_file, "r") as f:
                    self.session_data = json.load(f)
            logger.debug(f"Loaded session metadata from {self.session_file}")
        except Exception as e:
            logger.warning(f"Could not load session metadata: {e}")
            self.session_data = None

    def _update_session_file(self, action: str = "update") -> None:
        """Update session file with current state (multi-process safe)."""
        try:
            # Get Lobster version if available
            try:
                from importlib.metadata import version

                lobster_version = version("lobster")
            except Exception:
                lobster_version = "unknown"

            session_data = {
                "session_id": self.session_id,
                "created_at": (
                    self.session_data.get("created_at", datetime.now().isoformat())
                    if self.session_data
                    else datetime.now().isoformat()
                ),
                "last_modified": datetime.now().isoformat(),
                "lobster_version": lobster_version,
                "active_modalities": {},
                "workspace_stats": {
                    "total_datasets": len(self.available_datasets),
                    "total_loaded": len(self.modalities),
                    "total_size_mb": sum(
                        d["size_mb"] for d in self.available_datasets.values()
                    ),
                },
            }

            # Add loaded modalities info
            for name, adata in self.modalities.items():
                if name in self.available_datasets:
                    session_data["active_modalities"][name] = {
                        **self.available_datasets[name],
                        "last_accessed": datetime.now().isoformat(),
                    }
                else:
                    # For modalities not in available_datasets (newly created)
                    session_data["active_modalities"][name] = {
                        "shape": adata.shape,
                        "last_accessed": datetime.now().isoformat(),
                        "type": "in_memory",
                    }

            # Add command history from provenance if available
            if self.provenance and self.provenance.activities:
                # Keep last 50 activities
                recent_activities = self.provenance.activities[-50:]
                session_data["command_history"] = [
                    {"timestamp": act.get("timestamp"), "command": act.get("type")}
                    for act in recent_activities
                ]

            # Write atomically with inter-process lock
            with queue_file_lock(self._session_lock, self._session_lock_path):
                atomic_write_json(self.session_file, session_data)

            logger.debug(f"Updated session file: {self.session_file}")
        except Exception as e:
            logger.warning(f"Could not update session file: {e}")

    def load_dataset(self, name: str, force_reload: bool = False) -> bool:
        """Load a specific dataset from disk.

        Args:
            name: Name of the dataset to load
            force_reload: Whether to reload even if already in memory

        Returns:
            bool: True if successfully loaded, False otherwise
        """
        with self._measure_step("dm:load_dataset"):
            if name in self.modalities and not force_reload:
                return True  # Already loaded

            if name not in self.available_datasets:
                logger.error(f"Dataset '{name}' not found in workspace")
                return False

            try:
                path = Path(self.available_datasets[name]["path"])
                adata = _read_h5ad_smart(path, backed_mode_enabled=self.backed_mode)
                with self._state_lock:
                    old = self.modalities.get(name)
                    if old is not None:
                        _close_backed_handle(old)
                    self.modalities[name] = adata
                self._modality_sources[name] = path

                # Update session file
                self._update_session_file("loaded")

                # Log operation
                self.log_tool_usage(
                    tool_name="load_dataset",
                    parameters={"name": name},
                    description=(
                        f"Loaded dataset {name} ("
                        f"{self.available_datasets[name]['size_mb']:.1f} MB)"
                    ),
                )

                logger.info(f"Loaded dataset '{name}' from workspace")
                return True
            except HdfReadRefusal:
                raise
            except Exception as e:
                logger.error(f"Failed to load dataset {name}: {e}")
                return False

    def restore_session(
        self,
        pattern: str = "recent",
        max_size_mb: float = 1000,
        workspace_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Restore datasets based on pattern with memory limits.

        Args:
            pattern: Restoration pattern ('recent', 'all', or glob pattern)
            max_size_mb: Maximum total size of datasets to load
            workspace_name: Optional name of a saved workspace to restore from.
                When provided, loads modalities from the named workspace
                directory first, then applies the pattern filter to select
                which modalities to keep.

        Returns:
            Dict[str, Any]: Restoration results
        """
        restored = []
        skipped = []
        total_size = 0

        # If workspace_name is provided, load from that workspace first
        if workspace_name is not None:
            import fnmatch

            load_result = self.load_workspace(workspace_name=workspace_name)
            loaded_names = load_result.get("loaded", [])

            if pattern and pattern not in ("recent", "all"):
                # Apply glob pattern filter to loaded modalities
                matching = [
                    name for name in loaded_names if fnmatch.fnmatch(name, pattern)
                ]
                # Remove non-matching modalities
                for name in loaded_names:
                    if name not in matching:
                        with self._state_lock:
                            if name in self.modalities:
                                _close_backed_handle(self.modalities[name])
                                del self.modalities[name]
                        skipped.append((name, "pattern_mismatch"))
                restored = matching
            else:
                restored = loaded_names

            return {
                "restored": restored,
                "skipped": skipped,
                "total_size_mb": total_size,
                "pattern": pattern,
                "workspace_name": workspace_name,
            }

        if pattern == "recent":
            # Load datasets from last session
            if self.session_data and "active_modalities" in self.session_data:
                for name in self.session_data["active_modalities"]:
                    if name in self.available_datasets:
                        size_mb = self.available_datasets[name]["size_mb"]
                        if total_size + size_mb <= max_size_mb:
                            if self.load_dataset(name):
                                restored.append(name)
                                total_size += size_mb
                        else:
                            skipped.append((name, "size_limit"))

        elif pattern == "all":
            # Load all available datasets
            for name in sorted(self.available_datasets.keys()):
                size_mb = self.available_datasets[name]["size_mb"]
                if total_size + size_mb <= max_size_mb:
                    if self.load_dataset(name):
                        restored.append(name)
                        total_size += size_mb
                else:
                    skipped.append((name, "size_limit"))

        else:
            # Pattern matching (glob-style)
            import fnmatch

            for name in self.available_datasets:
                if fnmatch.fnmatch(name, pattern):
                    size_mb = self.available_datasets[name]["size_mb"]
                    if total_size + size_mb <= max_size_mb:
                        if self.load_dataset(name):
                            restored.append(name)
                            total_size += size_mb
                    else:
                        skipped.append((name, "size_limit"))

        return {
            "restored": restored,
            "skipped": skipped,
            "total_size_mb": total_size,
            "pattern": pattern,
        }

    # ========================================
    # NOTEBOOK PIPELINE SUPPORT
    # ========================================

    def export_notebook(
        self,
        name: str,
        description: str = "",
        filter_strategy: str = "successful",
    ) -> Path:
        """
        Export current session as Jupyter notebook.

        Args:
            name: Notebook filename (no extension)
            description: Human-readable description
            filter_strategy: "successful" | "all" | "manual" | "replay"

        Returns:
            Path to generated .ipynb file

        Raises:
            ValueError: If provenance tracking disabled or no activities

        Example:
            >>> path = dm.export_notebook(
            ...     name="standard_qc_workflow",
            ...     description="Quality control and clustering for 10X data"
            ... )
            >>> print(f"Exported: {path}")
            Exported: <workspace>/notebooks/standard_qc_workflow.ipynb
        """
        if not self.provenance:
            raise ValueError("Provenance tracking disabled - cannot export notebook")

        if not self.notebook_exporter:
            raise ValueError("Notebook exporter not initialized")

        if not self.provenance.activities:
            raise ValueError("No activities recorded - nothing to export")

        path = self.notebook_exporter.export(name, description, filter_strategy)

        logger.info(f"Notebook exported: {path}")
        logger.info("Next steps:")
        logger.info(f"  1. Review: jupyter notebook {path}")
        logger.info(f"  2. Commit: git add {path} && git commit -m 'Add {name}'")
        logger.info(f"  3. Run: /pipeline run {path.name} <modality>")

        return path

    def run_notebook(
        self,
        notebook_path: Union[str, Path],
        input_modality: str,
        parameters: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute saved notebook with new data.

        Args:
            notebook_path: Path to .ipynb file or filename in <workspace>/notebooks/
            input_modality: Name of modality to use as input
            parameters: Optional parameter overrides
            dry_run: If True, validate but don't execute

        Returns:
            Execution result with status, output_notebook, etc.

        Example:
            >>> # Dry run first
            >>> result = dm.run_notebook(
            ...     "standard_qc_workflow.ipynb",
            ...     "my_new_dataset",
            ...     dry_run=True
            ... )
            >>> print(result['validation'])

            >>> # Actually run
            >>> result = dm.run_notebook(
            ...     "standard_qc_workflow.ipynb",
            ...     "my_new_dataset"
            ... )
            >>> print(f"Output: {result['output_notebook']}")
        """
        # Resolve notebook path
        nb_path = Path(notebook_path)
        if not nb_path.exists():
            # Try workspace notebooks directory
            nb_path = self.workspace_path / "notebooks" / notebook_path

        if not nb_path.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")

        # Get input data
        if input_modality not in self.modalities:
            raise ValueError(f"Modality '{input_modality}' not loaded")

        # Save input modality as H5AD for notebook
        import tempfile

        temp_dir = Path(tempfile.mkdtemp())
        temp_input = temp_dir / "input.h5ad"
        self.save_modality(input_modality, temp_input)

        # Execute
        if dry_run:
            result = self.notebook_executor.dry_run(nb_path, temp_input)
        else:
            result = self.notebook_executor.execute(nb_path, temp_input, parameters)

        return result

    def list_notebooks(self) -> List[Dict[str, Any]]:
        """
        List available notebooks in workspace notebooks directory.

        Returns:
            List of notebook metadata
        """
        notebooks_dir = self.workspace_path / "notebooks"
        if not notebooks_dir.exists():
            return []

        notebooks = []
        for nb_file in notebooks_dir.glob("*.ipynb"):
            # Read metadata
            try:
                with open(nb_file) as f:
                    nbformat_module = _ensure_nbformat()
                    nb = nbformat_module.read(f, as_version=4)

                metadata = nb.metadata.get("lobster", {})

                notebooks.append(
                    {
                        "filename": nb_file.name,
                        "path": str(nb_file),
                        "name": nb_file.stem,
                        "created_by": metadata.get("created_by", "unknown"),
                        "created_at": metadata.get("created_at", ""),
                        "lobster_version": metadata.get("lobster_version", ""),
                        "n_steps": len([c for c in nb.cells if c.cell_type == "code"]),
                        "size_kb": nb_file.stat().st_size / 1024,
                    }
                )
            except Exception as e:
                logger.warning(f"Could not read notebook {nb_file}: {e}")

        return notebooks

    # ==================== PUBLICATION CACHING (Phase 2) ====================

    def cache_publication_content(
        self,
        identifier: str,
        content: Dict[str, Any],
        format: str = "markdown",
    ) -> Path:
        """
        Cache publication content for future retrieval.

        This method delegates to DoclingService for actual caching while providing
        a unified interface through DataManager. All publication caching should go
        through this method to ensure proper provenance tracking.

        Args:
            identifier: Publication identifier (PMID, DOI, or URL)
            content: Extraction result dictionary containing:
                - 'markdown': str - Extracted content as markdown
                - 'source': str - Source URL
                - 'parser': str - Parser used (docling/pypdf2)
                - 'methods_text': Optional[str] - Methods section text
                - 'software_detected': Optional[List[str]] - Detected software
            format: Cache format ('markdown' for human-readable, 'json' for structured)

        Returns:
            Path to cached file

        Raises:
            ManagedPathError: The cache file or its folder is inside the managed
                figure folder (nothing is written, no provenance is recorded)

        Example:
            >>> content = {
            ...     'markdown': '# Methods\\nWe used Seurat...',
            ...     'source': 'https://pmc.../PMC123.pdf',
            ...     'parser': 'docling',
            ...     'methods_text': 'Analysis with Seurat',
            ...     'software_detected': ['seurat']
            ... }
            >>> path = dm.cache_publication_content('PMID:12345678', content)
            >>> print(f"Cached to: {path}")

        Note:
            This is part of Phase 2 refactoring to consolidate all caching
            through DataManager (architectural requirement).
        """
        import hashlib

        publications_dir = self.literature_cache_dir / "publications"

        # Sanitize identifier for filename
        safe_identifier = (
            identifier.replace(":", "_").replace("/", "_").replace("\\", "_")
        )

        if format == "markdown":
            # Store as human-readable markdown
            cache_file = publications_dir / f"{safe_identifier}.md"
        elif format == "json":
            # Use MD5-based key for JSON (DoclingService format)
            source = content.get("source", identifier)
            cache_key = hashlib.md5(source.encode(), usedforsecurity=False).hexdigest()
            cache_file = publications_dir / f"{cache_key}.json"
        else:
            cache_file = None  # its ValueError still comes after the folder

        # The folder and the file itself, before the folder is created.
        _assert_dirs_allowed(self.workspace_path, publications_dir)
        if cache_file is not None:
            self._assert_save_allowed(cache_file)
        publications_dir.mkdir(parents=True, exist_ok=True)

        if format == "markdown":
            # Format markdown document
            markdown_content = f"""# Publication: {identifier}

**Extraction Date**: {content.get("timestamp", "N/A")}
**Parser**: {content.get("parser", "unknown")}

---

## Methods Section

{content.get("methods_text", "No methods section extracted")}

---

## Software Tools Detected

{chr(10).join(f"`{tool}`" for tool in content.get("software_detected", []))}

---

## Extraction Metadata

- **Source**: {content.get("source", "N/A")}
- **Parser**: {content.get("parser", "N/A")}
- **Fallback Used**: {content.get("fallback_used", False)}
- **Timestamp**: {content.get("timestamp", "N/A")}
"""

            self._assert_save_allowed(cache_file)
            cache_file.write_text(markdown_content, encoding="utf-8")
            logger.info(f"Cached publication as markdown: {cache_file.name}")

        elif format == "json":
            import json

            self._assert_save_allowed(cache_file)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2)

            logger.info(f"Cached publication as JSON: {cache_file.name}")

        else:
            raise ValueError(f"Unsupported cache format: {format}")

        # Log to provenance
        if self.provenance:
            self.log_tool_usage(
                tool_name="cache_publication_content",
                parameters={"identifier": identifier, "format": format},
                description=f"Cached publication content for {identifier}",
            )

        return cache_file

    def get_cached_publication(self, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached publication by identifier.

        This method checks for cached publication content in the literature cache
        directory. It supports both markdown and JSON formats.

        Args:
            identifier: Publication identifier (PMID, DOI, or URL)

        Returns:
            Dictionary with extraction results or None if not found

        Example:
            >>> content = dm.get_cached_publication('PMID:12345678')
            >>> if content:
            ...     print(f"Found cached content: {content['methods_text'][:100]}...")
            ... else:
            ...     print("No cache found, need to fetch")

        Note:
            This is part of Phase 2 refactoring. Delegates to existing
            cache infrastructure while providing unified DataManager interface.
        """
        import hashlib
        import json

        publications_dir = self.literature_cache_dir / "publications"

        # Sanitize identifier for filename
        safe_identifier = (
            identifier.replace(":", "_").replace("/", "_").replace("\\", "_")
        )

        # Try markdown format first
        markdown_file = publications_dir / f"{safe_identifier}.md"
        if markdown_file.exists():
            logger.info(f"Cache hit (markdown): {markdown_file.name}")
            markdown_content = markdown_file.read_text(encoding="utf-8")

            # Parse markdown to extract structured data
            result = {
                "identifier": identifier,
                "format": "markdown",
                "markdown": markdown_content,
                "cache_file": str(markdown_file),
                "cache_hit": True,
            }

            # Extract methods section if present
            if "## Methods Section" in markdown_content:
                methods_start = markdown_content.find("## Methods Section") + len(
                    "## Methods Section"
                )
                methods_end = markdown_content.find("---", methods_start)
                if methods_end > methods_start:
                    result["methods_text"] = markdown_content[
                        methods_start:methods_end
                    ].strip()

            # Extract software tools if present
            if "## Software Tools Detected" in markdown_content:
                software_start = markdown_content.find(
                    "## Software Tools Detected"
                ) + len("## Software Tools Detected")
                software_end = markdown_content.find("---", software_start)
                if software_end > software_start:
                    software_section = markdown_content[
                        software_start:software_end
                    ].strip()
                    # Extract tools from markdown code blocks
                    import re

                    tools = re.findall(r"`([^`]+)`", software_section)
                    result["software_detected"] = tools

            return result

        # Try JSON format (MD5-based key)
        cache_key = hashlib.md5(identifier.encode(), usedforsecurity=False).hexdigest()
        json_file = publications_dir / f"{cache_key}.json"
        if json_file.exists():
            logger.info(f"Cache hit (JSON): {json_file.name}")
            with open(json_file, "r", encoding="utf-8") as f:
                result = json.load(f)
            result["cache_hit"] = True
            result["cache_file"] = str(json_file)
            return result

        # No cache found
        logger.debug(f"Cache miss for identifier: {identifier}")
        return None

    def list_session_publications(self) -> List[Dict[str, Any]]:
        """
        List publications extracted in the current session.

        This method uses provenance tracking to identify publications that were
        extracted during the current session, then checks cache status for each.

        Returns:
            List of publication summaries:
            [{
                "identifier": str,
                "tool_name": str,
                "timestamp": str,
                "cache_status": "markdown" | "json" | "both" | "none",
                "methods_length": int,
                "source": str
            }]

        Examples:
            >>> publications = dm.list_session_publications()
            >>> for pub in publications:
            ...     print(f"{pub['identifier']}: {pub['cache_status']}")

        Note:
            This is part of Phase 3 migration. Session features will be
            fully refactored in Phase 4.
        """
        publications = []

        # Find all publication extraction operations in tool usage history
        extraction_tools = [
            "extract_pdf_content",
            "extract_methods_from_paper",
            "extract_methods_section",
            "extract_paper_methods",  # Phase 3 tool
        ]

        for entry in self.tool_usage_history:
            tool_name = entry.get("tool_name", "")
            if tool_name in extraction_tools:
                params = entry.get("parameters", {})
                description = entry.get("description", "")

                # Try to extract identifier from parameters
                identifier = (
                    params.get("url_or_pmid")
                    or params.get("source")
                    or params.get("url")
                    or "unknown"
                )

                # Truncate long URLs for display
                if len(identifier) > 80:
                    identifier = identifier[:77] + "..."

                # Check cache status
                cache_status = self._check_publication_cache_status(identifier)

                # Estimate methods length from description
                methods_length = 0
                if "characters" in description:
                    try:
                        methods_length = int(description.split()[0].replace(",", ""))
                    except (ValueError, IndexError):
                        pass

                publications.append(
                    {
                        "identifier": identifier,
                        "tool_name": tool_name,
                        "timestamp": entry.get("timestamp", "unknown"),
                        "cache_status": cache_status,
                        "methods_length": methods_length,
                        "source": params.get("parser", "unknown"),
                    }
                )

        logger.info(f"Found {len(publications)} publications in current session")
        return publications

    def _check_publication_cache_status(self, identifier: str) -> str:
        """
        Check cache status for a publication identifier.

        Args:
            identifier: Publication identifier

        Returns:
            Cache status: "markdown", "json", "both", or "none"
        """
        import hashlib

        has_markdown = False
        has_json = False

        # Check markdown cache
        safe_identifier = (
            identifier.replace(":", "_").replace("/", "_").replace("\\", "_")
        )
        publications_dir = self.literature_cache_dir / "publications"
        md_file = publications_dir / f"{safe_identifier}.md"

        if md_file.exists():
            has_markdown = True

        # Check JSON cache (MD5-based key)
        cache_key = hashlib.md5(identifier.encode(), usedforsecurity=False).hexdigest()
        json_file = publications_dir / f"{cache_key}.json"

        if json_file.exists():
            has_json = True

        # Return combined status
        if has_markdown and has_json:
            return "both"
        elif has_markdown:
            return "markdown"
        elif has_json:
            return "json"
        else:
            return "none"
