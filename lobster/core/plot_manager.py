"""
PlotManager: Extracted component for managing Plotly visualizations.

This module provides the PlotManager class for managing Plotly figure storage,
metadata tracking, and workspace export. Extracted from DataManagerV2 to
improve separation of concerns and testability.

All methods return 3-tuples: (result, stats, AnalysisStep|None) to follow
Lobster's standard service pattern and enable provenance tracking.
"""

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd

from lobster.core.provenance.analysis_ir import AnalysisStep

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

logger = logging.getLogger(__name__)

# Lazy-loaded plotly modules
_plotly_go = None
_plotly_io = None


def _ensure_plotly():
    """Lazily import plotly with a helpful error message."""
    global _plotly_go, _plotly_io
    if _plotly_go is None:
        try:
            import plotly.graph_objects as _plotly_go  # type: ignore
            import plotly.io as _plotly_io  # type: ignore
        except ImportError as exc:
            from lobster.core.component_registry import get_install_command

            cmd = get_install_command("plotly")
            raise ImportError(
                f"plotly is required for PlotManager operations. "
                f"Install with: {cmd}"
            ) from exc
    return _plotly_go, _plotly_io


def _one_line_reason(exc: BaseException, limit: int = 200) -> str:
    """First non-empty line of an exception message, bounded.

    Some exporters (Kaleido) put the reason on the line after a leading
    newline, which made the log and any relay look like an empty reason.
    Filesystem errors keep only their description and the file name, and
    any other message has absolute paths cut to their last component: the
    reason is shown to the model, which must not see absolute paths.
    """
    if isinstance(exc, OSError) and exc.filename is not None:
        # str(exc) would embed the full path; never fall back to it here.
        description = (exc.strerror or "").strip() or type(exc).__name__
        filename = exc.filename
        if isinstance(filename, bytes):
            filename = os.fsdecode(filename)
        name = Path(filename).name if isinstance(filename, str) else ""
        return (f"{description}: {name}" if name else description)[:limit]
    for line in str(exc).splitlines():
        line = line.strip().rstrip(",;:").strip()
        if line:
            return _strip_dirs(line)[:limit]
    return type(exc).__name__


# URLs are left alone (file: URLs are paths); only the text between them is
# scanned for paths. Paths containing spaces are only partly reduced.
_URL = re.compile(
    r"(?<![A-Za-z0-9+.-])(?![Ff][Ii][Ll][Ee]:)[A-Za-z][A-Za-z0-9+.-]*://\S+"
)

# An absolute path with at least one directory: POSIX ("/a/b"), Windows drive
# ("C:\\a\\b") or UNC ("\\\\host\\share\\b", "//host/share/b"). Relative paths ("plots/x.html",
# "~/x") and bare slashes ("1 /2") do not match. Segments exclude separators,
# so matching stays linear.
_ABSOLUTE_PATH = re.compile(
    r"(?<![\w.~\\/])(?:[A-Za-z]:[\\/]|\\\\|/{1,3})"
    r"(?:[^\s'\"<>|:;,()\[\]{}\\/]+[\\/])+"
    r"[^\s'\"<>|:;,()\[\]{}\\/]*"
)


def _last_component(path: str) -> str:
    parts = [p for p in re.split(r"[\\/]+", path) if p]
    return parts[-1] if parts else path


def _strip_dirs(text: str) -> str:
    """Replace absolute paths in an error message with their final component."""
    pieces = []
    last = 0
    for url in _URL.finditer(text):
        pieces.append(
            _ABSOLUTE_PATH.sub(
                lambda m: _last_component(m.group(0)), text[last : url.start()]
            )
        )
        pieces.append(url.group(0))
        last = url.end()
    pieces.append(
        _ABSOLUTE_PATH.sub(lambda m: _last_component(m.group(0)), text[last:])
    )
    return "".join(pieces)


class SuppressKaleidoLogging:
    """Context manager to suppress Kaleido's verbose logging during PNG export."""

    def __init__(self):
        self.original_level = logging.WARNING

    def __enter__(self):
        # Suppress kaleido and plotly logging
        logging.getLogger("kaleido").setLevel(logging.ERROR)
        logging.getLogger("plotly").setLevel(logging.ERROR)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original levels
        logging.getLogger("kaleido").setLevel(self.original_level)
        logging.getLogger("plotly").setLevel(self.original_level)
        return False


@dataclass(frozen=True)
class AttemptContext:
    """One producer invocation's attempt. Immutable; accepted only by the
    PlotManager that issued it, and only while its record is open (see ``begin_attempt``).
    ``fence`` is the store's born set captured at ``begin_attempt`` (None for a store
    without one); the commit uses the fence recorded at that time, never a later one."""

    owner: object = field(repr=False, compare=False)
    nonce: str
    fence: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class HeldInput:
    """A modality read under an attempt: the object actually used and its identity edge."""

    ctx_nonce: str
    name: str
    adata: Any = field(repr=False, compare=False)
    edge: Any


class PlotManager:
    """
    Manages Plotly figure storage, metadata tracking, and workspace export.

    This class is the single source of truth for plot management within
    a Lobster workspace. It handles:
    - Plot storage with comprehensive metadata
    - FIFO buffer with configurable max history
    - Thread-safe save operations with rate limiting
    - Visualization state tracking for agents
    - Workspace export (HTML/PNG)

    All public methods return 3-tuples: (result, stats, AnalysisStep|None)
    to follow Lobster's standard service pattern.

    Thread Safety:
        - `_save_lock`: Prevents concurrent saves
        - `_counter_lock`: Protects plot_counter increment
        - Rate limiting: Minimum 2s between saves (configurable)

    Attributes:
        workspace_path: Path to the workspace directory
        max_plots_history: Maximum number of plots to retain
        enable_ir: Whether to generate AnalysisStep IR for provenance
    """

    def __init__(
        self,
        workspace_path: Path,
        max_plots_history: int = 50,
        min_save_interval: float = 2.0,
        enable_ir: bool = True,
    ):
        """
        Initialize PlotManager.

        Args:
            workspace_path: Path to the workspace directory for plot export
            max_plots_history: Maximum number of plots to keep in history
            min_save_interval: Minimum seconds between save operations
            enable_ir: Whether to generate AnalysisStep IR for provenance
        """
        self.workspace_path = Path(workspace_path)
        self.max_plots_history = max_plots_history
        self.enable_ir = enable_ir

        # Plot storage
        self.latest_plots: List[Dict[str, Any]] = []
        self.plot_counter: int = 0
        self._counter_lock = threading.Lock()

        # Save safety mechanisms
        self._save_lock = threading.Lock()
        self._save_in_progress = False
        self._last_save_time = 0.0
        self._min_save_interval = min_save_interval

        # Managed figure lifecycle (contract §6.5): add_plot STAGES; commit_staged publishes
        # through the trusted saver after the owning tool's success path.
        from collections import OrderedDict

        self._staged: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        # (ref, path) pairs of published revisions; the drain returns the paths only
        self._committed_paths: List[Tuple[str, str]] = []
        self._figure_store_provider = None
        self._identity_resolver = None
        self._session_id_provider = None
        # Local writer-lock opt-in, OPTED plot managers only: ``_d7_root`` is the captured
        # existing writer RLock passed by an opted data manager (None = today's behaviour);
        # ``_d7_store`` is the ONE store bound at the first managed use; ``_d7_gen`` makes a
        # reconfiguration that races a first bind refuse it. Guarded by ``_attempt_lock``.
        self._d7_root = None
        self._d7_store = None
        self._d7_gen = 0
        # Attempts: nonce -> record (open | committing | retry_only | closed). One
        # lock guards the records and ctx-tagged staging; it is never held across a store
        # commit (no nesting with the store writer lock).
        self._attempt_owner = object()
        self._attempts: Dict[str, Dict[str, Any]] = {}
        self._attempt_lock = threading.RLock()

        # Visualization state management (for Visualization Expert Agent)
        self.visualization_state: Dict[str, Any] = {
            "history": [],  # List of created visualizations
            "settings": {  # User preferences
                "default_width": 800,
                "default_height": 600,
                "color_scheme": "Set1",
                "save_by_default": True,
                "export_formats": ["html", "png"],
            },
            "plot_registry": {},  # UUID -> plot metadata mapping
        }

        logger.debug(f"PlotManager initialized with workspace: {workspace_path}")

    # ========================================
    # CORE PLOT MANAGEMENT
    # ========================================

    def add_plot(
        self,
        plot: "Figure",
        title: Optional[str] = None,
        source: Optional[str] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        analysis_params: Optional[Dict[str, Any]] = None,
        modalities: Optional[Dict[str, Any]] = None,
        current_dataset: Optional[str] = None,
        *,
        context: Optional[AttemptContext] = None,
        inputs: Tuple[HeldInput, ...] = (),
        retained_source=None,
    ) -> Tuple[str, Dict[str, Any], Optional[AnalysisStep]]:
        """
        Add a plot to the collection with comprehensive metadata.

        Args:
            plot: Plotly Figure object
            title: Optional title for the plot
            source: Optional source identifier (e.g., service name)
            dataset_info: Optional information about the dataset used
            analysis_params: Optional parameters used for the analysis
            modalities: Optional dict of modalities for context (from DataManagerV2)
            current_dataset: Optional current dataset name for context

        Returns:
            Tuple of (plot_id, stats, AnalysisStep|None)

        Raises:
            ValueError: If plot is not a Plotly Figure
            AttemptContextError: ``context`` is not an open attempt of this manager, or an
                input was not read under it. With ``context`` the staged entry
                keeps exactly the held inputs' edges; nothing is resolved later by name.
        """
        from lobster.core.figures.errors import AttemptContextError

        if context is not None:
            with self._attempt_lock:
                self._attempt(context, ("open",))
            if not all(
                isinstance(h, HeldInput) and h.ctx_nonce == context.nonce
                for h in inputs
            ):
                raise AttemptContextError("an input was not read under this attempt")
        elif inputs:
            raise AttemptContextError("held inputs need their attempt context")
        try:
            go_module, _ = _ensure_plotly()
            if not isinstance(plot, go_module.Figure):
                raise ValueError("Plot must be a plotly Figure object.")

            # Generate unique identifier with thread safety
            with self._counter_lock:
                self.plot_counter += 1
                plot_id = f"plot_{self.plot_counter}"

            # Create timestamps
            timestamp = datetime.now().isoformat()
            human_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Build dataset context
            current_dataset_info = dataset_info or {}
            if modalities and not current_dataset_info:
                # Try current_dataset first, then first available
                target_name = current_dataset or (
                    list(modalities.keys())[0] if modalities else None
                )
                if target_name and target_name in modalities:
                    adata = modalities[target_name]
                    if hasattr(adata, "shape"):
                        current_dataset_info = {
                            "data_shape": adata.shape,
                            "modality_name": target_name,
                            "n_obs": (
                                adata.n_obs
                                if hasattr(adata, "n_obs")
                                else adata.shape[0]
                            ),
                            "n_vars": (
                                adata.n_vars
                                if hasattr(adata, "n_vars")
                                else adata.shape[1]
                            ),
                        }

            # Create enhanced title with context
            enhanced_title = title or "Untitled"
            if current_dataset_info and "modality_name" in current_dataset_info:
                modality_name = current_dataset_info["modality_name"]
                if modality_name not in enhanced_title:
                    enhanced_title = (
                        f"{enhanced_title} ({modality_name} - {human_timestamp})"
                    )
                else:
                    enhanced_title = f"{enhanced_title} ({human_timestamp})"
            elif current_dataset_info and "data_shape" in current_dataset_info:
                shape_info = f"{current_dataset_info['data_shape'][0]}x{current_dataset_info['data_shape'][1]}"
                enhanced_title = (
                    f"{enhanced_title} (Data: {shape_info} - {human_timestamp})"
                )
            else:
                enhanced_title = f"{enhanced_title} ({human_timestamp})"

            # The figure's own title is no longer mutated (contract §6.5); the enhanced
            # title is display metadata only.

            # Stage a byte-exact candidate from a deep copy (never visible until committed).
            from lobster.core.figures.payload import figure_to_bytes

            modality_name = (current_dataset_info or {}).get(
                "modality_name"
            ) or current_dataset
            # Display metadata for the legacy readers; it becomes visible ONLY on commit.
            plot_entry = {
                "id": plot_id,
                "title": enhanced_title,
                "original_title": title or "Untitled",
                "timestamp": timestamp,
                "human_timestamp": human_timestamp,
                "source": source or "unknown",
                "dataset_info": current_dataset_info,
                "analysis_params": analysis_params or {},
                "created_at": datetime.now(),
                "data_context": {
                    "has_modalities": bool(modalities),
                    "modality_names": list(modalities.keys()) if modalities else [],
                    "current_dataset": current_dataset,
                },
            }

            # Staged only: no public reader (latest_plots, get_plot_by_id, get_latest_plots,
            # get_plot_history, client/stream replies) sees it until commit_staged accepts it.
            staged_entry = {
                "figure_bytes": figure_to_bytes(plot),
                "retained_source": retained_source,
                "label": title or "Untitled",
                "source": source or "unknown",
                "modality_name": modality_name,
                "entry": plot_entry,
                "nonce": None if context is None else context.nonce,
                "edges": tuple(h.edge for h in inputs),
            }
            with self._attempt_lock:
                if context is not None:  # re-checked under the lock that guards it
                    rec = self._attempt(context, ("open",))
                    rec["sids"][plot_id] = {"state": "staged"}
                self._staged[plot_id] = staged_entry
                # bounded history: only context-free entries are evicted (an attempt's
                # entries are bounded by its own end_attempt)
                while len(self._staged) > self.max_plots_history:
                    victim = next(
                        (k for k, v in self._staged.items() if v.get("nonce") is None),
                        None,
                    )
                    if victim is None:
                        break
                    del self._staged[victim]

            # Build stats
            stats = {
                "plot_id": plot_id,
                "title": enhanced_title,
                "source": source or "unknown",
                "plots_in_history": len(self.latest_plots),
                "timestamp": timestamp,
                "status": "staged",
            }

            # Create IR if enabled
            ir = (
                self._create_add_plot_ir(
                    title=title,
                    source=source,
                    dataset_info=current_dataset_info,
                )
                if self.enable_ir
                else None
            )

            logger.info(
                f"Plot added: '{enhanced_title}' with ID {plot_id} from {source}"
            )

            return plot_id, stats, ir

        except (ValueError, AttemptContextError):
            raise
        except Exception as e:
            logger.exception(f"Error in add_plot: {e}")
            return "", {"error": str(e)}, None

    def clear_plots(self) -> Tuple[None, Dict[str, Any], Optional[AnalysisStep]]:
        """
        Clear all stored plots.

        Returns:
            Tuple of (None, stats, AnalysisStep|None)
        """
        import contextlib

        root = self._d7_root
        with root if root is not None else contextlib.nullcontext():  # opted: the root
            cleared_count = len(self.latest_plots)
            self.latest_plots = []

        stats = {
            "cleared_count": cleared_count,
            "timestamp": datetime.now().isoformat(),
        }

        ir = self._create_clear_plots_ir() if self.enable_ir else None

        logger.info(f"Cleared {cleared_count} plots")

        return None, stats, ir

    def get_plot_by_id(
        self, plot_id: str
    ) -> Tuple[Optional["Figure"], Dict[str, Any], None]:
        """
        Get a plot by its unique ID.

        Args:
            plot_id: The unique ID of the plot

        Returns:
            Tuple of (Figure|None, stats, None)
        """
        for plot_entry in self.latest_plots:
            if plot_entry["id"] == plot_id:
                stats = {
                    "found": True,
                    "plot_id": plot_id,
                    "title": plot_entry.get("title"),
                }
                return plot_entry["figure"], stats, None

        stats = {"found": False, "plot_id": plot_id}
        return None, stats, None

    def get_latest_plots(
        self, n: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], None]:
        """
        Get the n most recent plots with their metadata.

        Args:
            n: Number of plots to return (None for all)

        Returns:
            Tuple of (plot_entries, stats, None)
        """
        if n is None:
            plots = self.latest_plots
        else:
            plots = self.latest_plots[-n:]

        stats = {
            "total_plots": len(self.latest_plots),
            "returned_plots": len(plots),
            "requested": n,
        }

        return plots, stats, None

    def get_plot_history(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any], None]:
        """
        Get the complete plot history with minimal metadata (no figures).

        Returns:
            Tuple of (history_entries, stats, None)
        """
        history = [
            {
                "id": p["id"],
                "title": p["title"],
                "timestamp": p["timestamp"],
                "source": p["source"],
            }
            for p in self.latest_plots
        ]

        stats = {
            "total_plots": len(history),
            "timestamp": datetime.now().isoformat(),
        }

        return history, stats, None

    # ========================================
    # MANAGED FIGURE LIFECYCLE (contract §6.5)
    # ========================================

    def configure_figure_lifecycle(
        self,
        store_provider=None,
        identity_resolver=None,
        session_id_provider=None,
        *,
        d7_root=None,
    ) -> None:
        """Wire the figure store, dataset identity resolver and session id (DataManagerV2 does this).

        ``d7_root`` (local writer-lock opt-in): the captured existing writer RLock of an OPTED data
        manager. ``None`` (the default) keeps every existing behaviour. An opted plot manager
        binds ONE store at its first managed use (:meth:`_bound_store`) and then refuses any
        reconfiguration with ``ValueError`` (nothing changed).

        OFF -> OFF (non-opted now and after this call) keeps the original behaviour exactly:
        three plain assignments, no lock (a provider may reconfigure from inside a held
        ``_attempt_lock``). Any call that enables, keeps or disables the opt-in runs the
        short state update under ``_attempt_lock`` and bumps the bind generation; no provider
        is called and no root is entered. Enabling the opt-in is construction-before-exposure
        (the opted data manager configures its plot manager before handing it out).
        """
        if d7_root is None and self._d7_root is None:  # OFF -> OFF: unchanged
            self._figure_store_provider = store_provider
            self._identity_resolver = identity_resolver
            self._session_id_provider = session_id_provider
            return
        with self._attempt_lock:
            if self._d7_root is not None and self._d7_store is not None:
                raise ValueError(
                    "an opted plot manager cannot be reconfigured after its first managed use"
                )
            self._figure_store_provider = store_provider
            self._identity_resolver = identity_resolver
            self._session_id_provider = session_id_provider
            self._d7_root = d7_root
            self._d7_gen += 1

    def _bound_store(self, expected=None):
        """OPTED plot managers only: the ONE store this plot manager uses.

        The provider is called OUTSIDE ``_attempt_lock`` and no root is entered. The first
        managed use binds the resolved store only if its ``writer_lock`` IS the captured root
        (read, never acquired); every later resolution must return that SAME object (no
        chasing). ``expected`` (:meth:`forget_refs`): the resolved store must also be that
        object, checked before anything is bound. Any refusal is ``AttemptContextError``
        and changes nothing.
        """
        from lobster.core.figures.errors import AttemptContextError

        with self._attempt_lock:
            root, gen = self._d7_root, self._d7_gen
            provider = self._figure_store_provider
        if root is None:
            raise AttemptContextError("this plot manager is not opted in to LOCAL D7")
        if provider is None:
            raise AttemptContextError("an opted plot manager needs its store provider")
        store = provider()
        if expected is not None and store is not expected:
            raise AttemptContextError("not this plot manager's figure store")
        if getattr(store, "writer_lock", None) is not root:
            raise AttemptContextError(
                "the figure store is outside this plot manager's lock domain"
            )
        with self._attempt_lock:
            if self._d7_gen != gen:
                raise AttemptContextError(
                    "the plot manager was reconfigured during this use"
                )
            if self._d7_store is None:
                self._d7_store = store
            elif self._d7_store is not store:
                raise AttemptContextError("the figure store changed after first use")
        return store

    def _figure_store(self):
        if self._figure_store_provider is not None:
            return self._figure_store_provider()
        from lobster.core.figures.store import LocalFigureStore

        return LocalFigureStore(self.workspace_path, session_id=self._session_id())

    def _session_id(self) -> str:
        if self._session_id_provider is not None:
            value = self._session_id_provider()
            if value:
                return str(value)
        return "local"

    def _input_edges(self, modality_name: Optional[str]) -> tuple:
        if not modality_name:
            return ()
        if self._identity_resolver is not None:
            edge = self._identity_resolver(modality_name)
            if edge is not None:
                return (edge,)
        from lobster.core.figures.records import InputEdge

        return (InputEdge(modality_name, None, "declared", False),)

    def commit_staged(
        self,
        activity: Optional[Dict[str, Any]],
        *,
        tool_call_id: Optional[str],
        staged_ids: List[str],
        tool_name: Optional[str] = None,
        agent: Optional[str] = None,
        context: Optional[AttemptContext] = None,
    ):
        """Publish staged figures through the trusted saver (call AFTER ``log_tool_usage``).

        Must be the owning tool's last fallible step. Returns ``List[CommitResult]`` (one per
        staged id, in order) and raises :class:`FigureError` only if none became available.
        Inside a graph run ``tool_call_id`` is mandatory (contract §6.2).

        ``context`` (attempt): commit SIDs staged under that open attempt, with their held
        edges and the attempt's ORIGINAL fence (see :meth:`_commit_in_context`). Without
        ``context`` a SID staged under an attempt is refused (``attempt_context_missing``);
        the context-free path is NOT race-safe (commit-time edges by name). An OPTED plot
        manager uses its ONE bound store and the gated publication on both paths
        (:meth:`_bound_store`, :meth:`_publish_result`); a non-opted one is unchanged.
        """
        from lobster.core.figures.errors import AttemptContextError, FigureError
        from lobster.core.figures.saver import TrustedSaver, outcome_state, summarize

        if not tool_call_id:
            raise AttemptContextError(
                "commit_staged requires the tool_call_id of the owning tool call"
            )
        if context is not None:
            return self._commit_in_context(
                context, activity, tool_call_id, staged_ids, tool_name, agent
            )
        missing = [sid for sid in staged_ids if sid not in self._staged]
        if missing:
            raise FigureError(
                f"unknown or already-committed staged figure(s): {missing}",
                details={"reason": "staged_figure_unknown"},
            )
        entries = [(sid, self._staged[sid]) for sid in staged_ids]
        if any(entry.get("nonce") is not None for _, entry in entries):
            raise AttemptContextError(
                "a figure staged under an attempt needs its context"
            )
        candidates, info = self._attempt_candidates(
            entries,
            base=f"{self._session_id()}:{tool_call_id}",
            activity=activity,
            tool_name=tool_name,
            agent=agent,
            inputs_of=lambda entry: self._input_edges(entry["modality_name"]),
        )
        # Opted: ONE bound store for this call; non-opted: resolved as before.
        bound = self._bound_store() if self._d7_root is not None else None
        results = TrustedSaver(
            bound if bound is not None else self._figure_store()
        ).commit_attempt(info, candidates, succeeded=True, fence=None)
        for sid, _ in entries:  # every outcome is known before any legacy publish
            self._staged.pop(sid, None)
        for (sid, staged), cand, res in zip(entries, candidates, results):
            if res.status == "available":
                self._publish_result(bound, staged, res, cand)

        if outcome_state(results) == "none":  # pending is not a failure (contract §6.3)
            raise FigureError(
                summarize(results),  # built from stable reason codes only
                details={
                    "reason": "no_figure_available",
                    "results": [r.to_dict() for r in results],
                },
            )
        return results

    def _attempt_candidates(
        self, entries, *, base, activity, tool_name, agent, inputs_of
    ) -> tuple:
        """(candidates, ExecutionInfo) for one commit batch; ``inputs_of(entry)`` gives
        each candidate's input edges. Index = position in the batch."""
        from lobster.core.figures.records import Candidate, ExecutionInfo

        candidates = [
            Candidate(
                attempt_key=f"{base}:{index}",
                candidate_index=index,
                figure_bytes=entry["figure_bytes"],
                config_bytes=b"{}",
                actor=f"legacy_wrapper:{entry['source']}",
                label=entry["label"],
                inputs=inputs_of(entry),
                source=entry.get("retained_source"),
            )
            for index, (_, entry) in enumerate(entries)
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
            tool=tool_name or entries[0][1]["source"],
            agent=agent,
            parameters={},
            persist=True,
            code_bytes=None,
            activity=activity,
            engine_version=engine_version,
            plotly_version=plotly_version,
        )
        return candidates, info

    # ------------------------------------------------------------------ attempts
    def begin_attempt(self) -> AttemptContext:
        """Start one producer invocation's attempt. Call it FIRST, before any input
        read: it captures the store's born-set fence (None for a store without one)."""
        import secrets

        opted = self._d7_root is not None
        store = self._bound_store() if opted else self._figure_store()
        capture = getattr(store, "attempt_fence", None)
        fence = capture() if callable(capture) else None
        nonce = secrets.token_hex(16)
        rec = {"state": "open", "fence": fence, "sids": {}, "batch": None}
        if opted:
            rec["store"] = store  # this attempt's ONE store (opted only)
        with self._attempt_lock:
            self._attempts[nonce] = rec
        return AttemptContext(self._attempt_owner, nonce, fence)

    def _attempt(self, context, states) -> Dict[str, Any]:
        """The live record of ``context`` in one of ``states`` (caller holds
        ``_attempt_lock``); anything else -> AttemptContextError."""
        from lobster.core.figures.errors import AttemptContextError

        rec = (
            self._attempts.get(context.nonce)
            if isinstance(context, AttemptContext)
            and context.owner is self._attempt_owner
            else None
        )
        if (
            rec is None
            or rec["state"] not in states
            or context.fence is not rec["fence"]
        ):
            raise AttemptContextError("this attempt context is not usable here")
        return rec

    def read_input(self, context: AttemptContext, data_manager, name: str) -> HeldInput:
        """Read one modality for this attempt: the object actually used and its identity edge,
        taken together under the store writer lock, so a deletion (which takes the same lock)
        cannot interleave between them. Not atomic provider proof: the provider branch still
        resolves by name, and a same-name replacement does not take the lock (stated limits).
        Lock order: store writer lock -> the data manager's state lock."""
        import contextlib

        from lobster.core.figures.errors import AttemptContextError
        from lobster.core.figures.records import InputEdge

        with self._attempt_lock:
            bound = self._attempt(context, ("open",)).get("store")
        if getattr(data_manager, "plot_manager", None) is not self:
            raise AttemptContextError(
                "read_input needs this plot manager's data manager"
            )
        store = bound if bound is not None else self._figure_store()
        lock = getattr(store, "writer_lock", None)
        with lock if lock is not None else contextlib.nullcontext():
            adata = data_manager.get_modality(name)
            edge = data_manager.input_identity_of(name, adata)
        if edge is None:
            edge = InputEdge(name, None, "declared", False)
        return HeldInput(context.nonce, name, adata, edge)

    def _commit_in_context(
        self, context, activity, tool_call_id, staged_ids, tool_name, agent
    ):
        """One commit batch per attempt (the execution plan binds the candidate count and
        every (index, digest), so a retry must resubmit the SAME batch).

        First call: SIDs staged under ``context``, unique, non-empty. Retry (state
        ``retry_only``): exactly the same SIDs in the same order and the same
        ``tool_call_id``; the store acknowledges already-available members (same claims and
        refs, never a second revision). Edges are the held ones from staging; the fence is
        the one recorded at ``begin_attempt``. Every returned outcome is recorded BEFORE any
        legacy publish. Afterwards: any pending (or a raise mid-call) -> ``retry_only``,
        else ``closed`` (stays closed).
        """
        from lobster.core.figures.errors import AttemptContextError, FigureError
        from lobster.core.figures.records import STATUS_AVAILABLE, STATUS_PENDING
        from lobster.core.figures.saver import TrustedSaver, outcome_state, summarize

        sids = tuple(staged_ids)
        with self._attempt_lock:
            rec = self._attempt(context, ("open", "retry_only"))
            bound = rec.get("store")  # opted only: the attempt's ONE store
            prior = rec["state"]
            batch = rec["batch"]
            if batch is None:
                if (
                    not sids
                    or len(set(sids)) != len(sids)
                    or any(
                        rec["sids"].get(sid, {}).get("state") != "staged"
                        or sid not in self._staged
                        for sid in sids
                    )
                ):
                    raise AttemptContextError("not this attempt's staged figures")
                batch = {
                    "sids": sids,
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "agent": agent,
                    "base": f"{self._session_id()}:{tool_call_id}",
                }
            elif sids != batch["sids"] or tool_call_id != batch["tool_call_id"]:
                raise AttemptContextError("a retry must resubmit the same batch")
            entries = [(sid, self._staged[sid]) for sid in sids]
            # BUSY from here until this call returns or raises, i.e. through publication and
            # result delivery: a concurrent commit/retry, add or read is refused (_attempt),
            # and so is a concurrent end_attempt. No other state is reachable.
            rec["state"] = "committing"
        final = prior  # the state left behind if nothing was submitted
        try:
            candidates, info = self._attempt_candidates(
                entries,
                base=batch["base"],
                activity=activity,
                tool_name=batch["tool_name"],
                agent=batch["agent"],
                inputs_of=lambda entry: entry["edges"],
            )
            with self._attempt_lock:
                rec["batch"] = batch
                for index, sid in enumerate(sids):
                    state = rec["sids"][sid]
                    state["index"] = index
                    state["delivered"] = False
                    if state["state"] != STATUS_AVAILABLE:
                        state["state"] = "submitted"  # BEFORE the call
                final = "retry_only"  # something may be submitted from here on
            results = TrustedSaver(
                bound if bound is not None else self._figure_store()
            ).commit_attempt(info, candidates, succeeded=True, fence=rec["fence"])
            with self._attempt_lock:  # ALL outcomes recorded before any publish
                for sid, res in zip(sids, results):
                    rec["sids"][sid].update(state=res.status, result=res)
                uncertain = any(
                    rec["sids"][sid]["state"] in (STATUS_PENDING, "submitted")
                    for sid in sids
                )
                final = "retry_only" if uncertain else "closed"
            for (sid, staged), cand, res in zip(entries, candidates, results):
                with self._attempt_lock:
                    state = rec["sids"][sid]
                    publish = res.status == STATUS_AVAILABLE and not state.get(
                        "published"
                    )
                    if publish:
                        state["published"] = True  # never twice (a retry re-ACKs)
                if publish:  # _attempt_lock NOT held here
                    self._publish_result(bound, staged, res, cand)
            with self._attempt_lock:
                for sid in sids:
                    rec["sids"][sid]["delivered"] = True
            if outcome_state(results) == "none":  # pending is not a failure (§6.3)
                raise FigureError(
                    summarize(results),
                    details={
                        "reason": "no_figure_available",
                        "results": [r.to_dict() for r in results],
                    },
                )
            return results
        finally:  # leave BUSY only now; closed stays closed
            with self._attempt_lock:
                rec["state"] = final
                if final == "closed":
                    for sid in rec["sids"]:
                        self._staged.pop(sid, None)

    def end_attempt(self, context: AttemptContext) -> Dict[str, Any]:
        """Close this attempt; call it in the producer's ``finally`` (after its own commit
        returned or raised). A closed attempt stays closed: every later read/add/commit with
        this context is refused. Refused (typed) while a commit of this attempt is still
        running in another thread, and for a context whose fence was swapped.

        Returns ``{sid: CommitResult}`` for the outcomes the producer has NOT received:
        a SID never submitted -> ``not_committed``/``attempt_failed`` (no claim was ever
        written; ``candidate_index`` -1 when it was in no batch); a SID whose commit call
        raised -> ``pending``/``unexpected_error`` (outcome unknown; never
        ``not_committed``); a recorded result whose call did not return (e.g. a legacy
        publish failure) -> that result. Delivered results are not repeated.
        """
        from lobster.core.figures.errors import AttemptContextError
        from lobster.core.figures.records import (
            STATUS_NOT_COMMITTED,
            STATUS_PENDING,
            CommitResult,
        )

        with self._attempt_lock:
            if not (
                isinstance(context, AttemptContext)
                and context.owner is self._attempt_owner
            ):
                raise AttemptContextError("not an attempt of this plot manager")
            rec = self._attempts.get(context.nonce)
            if rec is None:
                return {}
            if context.fence is not rec["fence"]:
                raise AttemptContextError("this attempt context is not usable here")
            if rec["state"] == "committing":  # never claim not_committed while it runs
                raise AttemptContextError("this attempt is still committing")
            del self._attempts[context.nonce]
            rec["state"] = "closed"
            out: Dict[str, Any] = {}
            for sid, state in rec["sids"].items():
                entry = self._staged.pop(sid, None)
                label = entry["label"] if entry else None
                if state["state"] == "staged":
                    out[sid] = CommitResult(
                        state.get("index", -1),
                        STATUS_NOT_COMMITTED,
                        reason="attempt_failed",
                        label=label,
                    )
                elif state["state"] == "submitted":
                    out[sid] = CommitResult(
                        state["index"],
                        STATUS_PENDING,
                        reason="unexpected_error",
                        label=label,
                    )
                elif not state.get("delivered", True):
                    out[sid] = state["result"]
            return out

    def _publish_result(self, bound, staged: Dict[str, Any], res, candidate) -> None:
        """Legacy-visible publication of one AVAILABLE result (call with ``_attempt_lock``
        NOT held).

        Non-opted (``bound is None``): exactly as before (the path is queued, then the entry
        is published). Opted: the entry is prepared with no lock; then,
        under the bound store's root, ``read_record`` success appends the ``(ref, path)``
        pair and then the entry in ONE hold. A ``_prepare_entry`` failure (raised) or a
        failed ``read_record`` (a deletion hid it meanwhile; not published) leaves BOTH
        containers unchanged. This is not a transaction: ``_append_entry`` can still raise
        after the pair was queued (no rollback). A METADATA-only presence gate, NOT
        payload-integrity revalidation. The caller's CommitResult and per-SID outcomes are
        unchanged (a raise here leaves the recorded result to ``end_attempt``).
        """
        if bound is None:
            self._committed_paths.append((res.ref, res.path))
            self._publish_entry(staged, res, candidate)
            return
        entry = self._prepare_entry(staged, res, candidate)
        with bound.writer_lock:
            try:
                bound.read_record(res.ref)
            except Exception:  # hidden, absent, pending or unreadable: not published
                return
            self._committed_paths.append((res.ref, res.path))
            self._append_entry(entry)

    def _publish_entry(self, staged: Dict[str, Any], res, candidate) -> None:
        """Make an ACCEPTED revision visible to the legacy readers (non-opted path).

        ``figure`` is rebuilt from the committed bytes (one truth), not the caller's object.
        """
        self._append_entry(self._prepare_entry(staged, res, candidate))

    def _prepare_entry(self, staged: Dict[str, Any], res, candidate) -> Dict[str, Any]:
        go_module, _ = _ensure_plotly()
        entry = dict(staged["entry"])
        entry["figure"] = go_module.Figure(json.loads(candidate.figure_bytes))
        entry["figure_ref"] = res.ref
        entry["figure_path"] = res.path
        return entry

    def _append_entry(self, entry: Dict[str, Any]) -> None:
        self.latest_plots.append(entry)
        if len(self.latest_plots) > self.max_plots_history:
            self.latest_plots.pop(0)

    def forget_refs(self, store, refs) -> frozenset:
        """OPTED plot managers only: drop exactly these revisions from BOTH
        legacy containers, under the captured root (re-entrant).

        ALREADY BOUND (the case inside a deletion's outer root hold): ``store`` must BE the
        bound store (immutable once bound), checked directly; then only the captured root
        is taken. No provider call and no ``_attempt_lock``, so a changed provider does not
        stop forgetting from this cache, and any other store is refused
        (``AttemptContextError``). Live data manager / provider eligibility is the deletion
        planner's job, not this method's.

        NOT YET BOUND (standalone only): the first bind runs through :meth:`_bound_store`
        (provider and ``_attempt_lock``) BEFORE this call's own root, without generating a
        figure. It must NOT be performed inside an outer root hold: a caller holding the
        root binds first (e.g. :meth:`_bound_store` before its hold) and then calls this.

        ``refs`` are revision ref strings; a malformed one raises ``ValueError`` before any
        container changes. ``latest_plots`` entries match by
        ``figure_ref``; queued pairs match by ref or by ``store.figure_path(ref)`` (so an
        entry already evicted from the history is still forgotten). Returns the UNIQUE refs
        actually removed (empty if none). ``latest_plots`` is replaced by a filtered list,
        as :meth:`clear_plots` does, so a concurrent reader sees either state; lists and
        figures already returned are not revoked (documented limit).
        """
        from lobster.core.figures.errors import AttemptContextError

        bound = self._d7_store  # set once, never rebound (reconfiguration is refused)
        if bound is None:
            bound = self._bound_store(expected=store)  # standalone first bind only
        elif store is not bound:
            raise AttemptContextError("not this plot manager's figure store")
        wanted = frozenset(refs)
        by_path = {bound.figure_path(ref): ref for ref in wanted}  # pure
        removed = set()
        with self._d7_root:  # the captured root: IS bound.writer_lock (checked at bind)
            kept_plots = []
            for entry in self.latest_plots:
                ref = entry.get("figure_ref")
                if ref is not None and ref in wanted:
                    removed.add(ref)
                else:
                    kept_plots.append(entry)
            kept_pairs = []
            for ref, path in self._committed_paths:
                hit = ref if ref in wanted else by_path.get(path)
                if hit is not None:
                    removed.add(hit)
                else:
                    kept_pairs.append((ref, path))
            if len(kept_plots) != len(self.latest_plots):
                self.latest_plots = kept_plots
            self._committed_paths[:] = kept_pairs
        return frozenset(removed)

    def discard_staged(self, staged_ids: Optional[List[str]] = None) -> int:
        """Drop staged (never visible) figures; a preview is not a revision."""
        if staged_ids is None:
            count = len(self._staged)
            self._staged.clear()
            return count
        return sum(1 for sid in staged_ids if self._staged.pop(sid, None) is not None)

    def save_plots_to_workspace(
        self,
    ) -> Tuple[List[str], Dict[str, Any], Optional[AnalysisStep]]:
        """Return paths of figures committed since the last call. Writes nothing.

        Figures are saved only through the trusted save step (``commit_staged`` / custom-code
        ``submit_figure``); this method no longer rewrites HTML/PNG copies.
        """
        import contextlib

        root = self._d7_root
        with root if root is not None else contextlib.nullcontext():  # opted: the root
            pairs = list(self._committed_paths)
            self._committed_paths.clear()
        paths = [path for _, path in pairs]  # paths only, as before
        return paths, {"saved_count": len(paths), "mode": "trusted_save_paths"}, None

    # ========================================
    # VISUALIZATION STATE MANAGEMENT
    # ========================================

    def add_visualization_record(
        self, plot_id: str, metadata: Dict[str, Any]
    ) -> Tuple[None, Dict[str, Any], None]:
        """
        Track visualization creation in the visualization state.

        Args:
            plot_id: Unique identifier for the plot
            metadata: Metadata about the visualization

        Returns:
            Tuple of (None, stats, None)
        """
        try:
            # Add to history
            self.visualization_state["history"].append(
                {
                    "plot_id": plot_id,
                    "timestamp": pd.Timestamp.now(),
                    "metadata": metadata,
                }
            )

            # Add to registry
            self.visualization_state["plot_registry"][plot_id] = metadata

            stats = {
                "plot_id": plot_id,
                "history_size": len(self.visualization_state["history"]),
                "registry_size": len(self.visualization_state["plot_registry"]),
            }

            logger.debug(f"Added visualization record for plot {plot_id}")

            return None, stats, None

        except Exception as e:
            logger.error(f"Failed to add visualization record: {e}")
            return None, {"error": str(e)}, None

    def get_visualization_history(
        self, limit: int = 10
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], None]:
        """
        Get recent visualization history.

        Args:
            limit: Maximum number of records to return

        Returns:
            Tuple of (history, stats, None)
        """
        try:
            history = self.visualization_state["history"][-limit:]
            stats = {
                "total_history": len(self.visualization_state["history"]),
                "returned": len(history),
                "limit": limit,
            }
            return history, stats, None
        except Exception as e:
            logger.error(f"Failed to get visualization history: {e}")
            return [], {"error": str(e)}, None

    def get_visualization_settings(self) -> Tuple[Dict[str, Any], Dict[str, Any], None]:
        """
        Get current visualization settings.

        Returns:
            Tuple of (settings, stats, None)
        """
        settings = self.visualization_state["settings"].copy()
        stats = {"settings_count": len(settings)}
        return settings, stats, None

    def update_visualization_settings(
        self, settings: Dict[str, Any]
    ) -> Tuple[None, Dict[str, Any], None]:
        """
        Update visualization settings.

        Args:
            settings: Settings to update

        Returns:
            Tuple of (None, stats, None)
        """
        try:
            self.visualization_state["settings"].update(settings)
            stats = {
                "updated_keys": list(settings.keys()),
                "timestamp": datetime.now().isoformat(),
            }
            logger.debug(f"Updated visualization settings: {settings}")
            return None, stats, None
        except Exception as e:
            logger.error(f"Failed to update visualization settings: {e}")
            return None, {"error": str(e)}, None

    def get_plot_by_uuid(
        self, plot_id: str
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], None]:
        """
        Get plot metadata by UUID from visualization registry.

        Args:
            plot_id: UUID of the plot

        Returns:
            Tuple of (metadata|None, stats, None)
        """
        metadata = self.visualization_state["plot_registry"].get(plot_id)
        stats = {
            "found": metadata is not None,
            "plot_id": plot_id,
        }
        return metadata, stats, None

    def clear_visualization_history(self) -> Tuple[None, Dict[str, Any], None]:
        """
        Clear visualization history and registry.

        Returns:
            Tuple of (None, stats, None)
        """
        cleared_history = len(self.visualization_state["history"])
        cleared_registry = len(self.visualization_state["plot_registry"])

        self.visualization_state["history"] = []
        self.visualization_state["plot_registry"] = {}

        stats = {
            "cleared_history": cleared_history,
            "cleared_registry": cleared_registry,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info("Cleared visualization history and registry")

        return None, stats, None

    # ========================================
    # PROVENANCE IR HELPERS
    # ========================================

    def _create_add_plot_ir(
        self,
        title: Optional[str],
        source: Optional[str],
        dataset_info: Dict[str, Any],
    ) -> AnalysisStep:
        """Create provenance IR for add_plot operation."""
        return AnalysisStep(
            operation="plot_manager.add_plot",
            tool_name="PlotManager.add_plot",
            description=f"Add plot '{title or 'Untitled'}' from {source or 'unknown'}",
            library="plotly",
            imports=[
                "from lobster.core.plot_manager import PlotManager",
                "import plotly.graph_objects as go",
            ],
            code_template="""# Add plot to PlotManager
# (Plot figure created by analysis service)
plot_id, stats, ir = plot_manager.add_plot(
    plot=fig,
    title={{ title | string | repr }},
    source={{ source | string | repr }},
)
print(f"Added plot: {plot_id}")
""",
            parameters={
                "title": title,
                "source": source,
                "dataset_info": dataset_info,
            },
            parameter_schema={
                "title": {
                    "type": "string",
                    "optional": True,
                    "description": "Title for the plot",
                },
                "source": {
                    "type": "string",
                    "optional": True,
                    "description": "Source identifier (service name)",
                },
            },
            input_entities=["plot_figure"],
            output_entities=["plot_id"],
        )

    def _create_clear_plots_ir(self) -> AnalysisStep:
        """Create provenance IR for clear_plots operation."""
        return AnalysisStep(
            operation="plot_manager.clear_plots",
            tool_name="PlotManager.clear_plots",
            description="Clear all stored plots",
            library="lobster",
            imports=["from lobster.core.plot_manager import PlotManager"],
            code_template="""# Clear all plots
_, stats, ir = plot_manager.clear_plots()
print(f"Cleared {stats['cleared_count']} plots")
""",
            parameters={},
            parameter_schema={},
            input_entities=[],
            output_entities=[],
        )

    def _create_save_plots_ir(self, saved_count: int) -> AnalysisStep:
        """Create provenance IR for save_plots_to_workspace operation."""
        return AnalysisStep(
            operation="plot_manager.save_plots_to_workspace",
            tool_name="PlotManager.save_plots_to_workspace",
            description=f"Save {saved_count} plots to workspace",
            library="plotly",
            imports=[
                "from lobster.core.plot_manager import PlotManager",
            ],
            code_template="""# Save plots to workspace
saved_files, stats, ir = plot_manager.save_plots_to_workspace()
print(f"Saved {len(saved_files)} files to {stats['workspace_path']}")
""",
            parameters={"saved_count": saved_count},
            parameter_schema={},
            input_entities=["latest_plots"],
            output_entities=["saved_files"],
        )
