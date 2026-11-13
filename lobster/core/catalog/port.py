"""The catalog port: the only interface Lobster core uses to talk to a catalog.

An engine implements ``Catalog`` synchronously and may raise; core code never calls
an engine directly. It goes through ``ThreadedCatalog`` (one worker thread, timeout,
fail-open) or ``NullCatalog``. Paths are absolute local paths; keys are POSIX paths
relative to the workspace root (``data/x.h5ad``).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
)

RunStatus = Literal["completed", "errored", "aborted"]


@dataclass(frozen=True)
class SessionInfo:
    """The Lobster session a catalog run stands for."""

    session_id: str
    workspace_path: Path
    lobster_version: str
    session_dir: Optional[Path] = None
    kind: str = "session"  # "session" (a client session) or "command" (one invocation)


@dataclass(frozen=True)
class ArtifactRef:
    """A file recorded in the catalog.

    ``path`` is the file the record points at. ``existing`` is True when identical
    content was already recorded under the key: no new version was created and
    ``path`` is that earlier version's file.
    """

    uid: str
    key: str
    path: Path
    version: Optional[str] = None
    existing: bool = False


@dataclass(frozen=True)
class TransformRef:
    """Code recorded in the catalog (an exported notebook)."""

    uid: str
    key: str


@dataclass(frozen=True)
class CatalogStatus:
    """What ``lobster lamin status`` shows."""

    engine: str
    mode: Literal["off", "workspace", "instance"]
    instance: Optional[str] = None
    location: Optional[str] = None
    run: Optional[str] = None
    degraded: bool = False
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecordedSave:
    """One save being recorded: where to keep the old version, and how to register.

    ``recorded`` is False when the save goes ahead without a record (the catalog
    was busy or is degraded); ``register`` then does nothing.
    """

    keep: Optional[Path] = None
    recorded: bool = True
    _register: Optional[Callable[..., Optional[ArtifactRef]]] = field(
        default=None, repr=False
    )

    def __init__(
        self,
        keep: Optional[Path] = None,
        recorded: bool = True,
        register: Optional[Callable[..., Optional[ArtifactRef]]] = None,
    ):
        self.keep = keep
        self.recorded = recorded and register is not None
        self._register = register

    def register(self, **fields: Any) -> Optional[ArtifactRef]:
        """Record the written file (``key``, ``kind``, ``otype``, ``description``, ``meta``)."""
        if not self.recorded or self._register is None:
            return None
        return self._register(**fields)


@contextmanager
def plain_save_section(catalog: "Catalog", path: Path) -> Iterator[RecordedSave]:
    """``save_section`` for a catalog without one (an engine used directly): no lock."""
    path = Path(path)
    try:
        keep = catalog.keep_path(path)
    except Exception:  # fail-open: the save goes ahead, still recorded
        keep = None
    yield RecordedSave(
        keep=keep, register=lambda **fields: catalog.register_artifact(path, **fields)
    )


@runtime_checkable
class Catalog(Protocol):
    """Record what a Lobster session saves, loads and exports."""

    def begin_run(self, session: SessionInfo) -> None:
        """Start the run for ``session``; a run still open is ended as completed first."""

    def keep_path(self, path: Path) -> Optional[Path]:
        """Where to keep the recorded version at ``path`` before new bytes replace it.

        The file name always holds the latest bytes, as without a catalog. When
        ``path`` holds a recorded version that is still unchanged, the engine answers
        a free path inside the workspace; the caller moves the file there right
        before it writes, and the next ``register_artifact`` of ``path`` points the
        record at the kept copy. None: nothing to keep, write over ``path``.
        The engine changes no file here, so an answer that comes too late is
        harmless: the caller has not moved anything.
        """

    def register_artifact(
        self,
        path: Path,
        *,
        key: str,
        kind: str,
        otype: Optional[str],
        description: str,
        meta: Dict[str, Any],
    ) -> Optional[ArtifactRef]:
        """Record the file at ``path`` as a new version of ``key``, output of the run.

        ``existing=True``: this content was already recorded; ``ArtifactRef.path``
        is the record's file. A save that was not prepared (the catalog was still
        starting, or ``keep_path`` answered late) may have rewritten a recorded
        file; the engine must not let a recorded version silently point at changed
        bytes.
        """

    def note_unrecorded(self, path: Path) -> None:
        """A save wrote ``path`` without being recorded (the catalog was busy).

        Records whose bytes the file no longer holds are marked as superseded,
        so no record silently points at changed bytes. Records nothing new.
        """

    def link_input(self, path: Path) -> Optional[ArtifactRef]:
        """Mark a recorded file with the same content as ``path`` as input of the run."""

    def register_transform(
        self, *, key: str, kind: str, source: str, description: str
    ) -> Optional[TransformRef]:
        """Record exported code (``kind="notebook"``) created by the run."""

    def end_run(
        self,
        status: RunStatus,
        report_path: Optional[Path] = None,
        environment_path: Optional[Path] = None,
    ) -> None:
        """Close the open run, if any."""

    def status(self) -> CatalogStatus:
        """Mode, location and health of the catalog."""
