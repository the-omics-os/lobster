"""The default catalog: records nothing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from lobster.core.catalog.port import (
    ArtifactRef,
    CatalogStatus,
    RunStatus,
    SessionInfo,
    TransformRef,
)


class NullCatalog:
    """Every call is a no-op; saves land where they were aimed.

    ``reason`` says why there is no catalog (shown by ``status``), e.g. that the
    user enabled it but the engine is not installed.
    """

    def __init__(self, reason: Optional[str] = None):
        self.reason = reason

    def begin_run(self, session: SessionInfo) -> None:
        return None

    def keep_path(self, path: Path) -> Optional[Path]:
        return None

    def note_unrecorded(self, path: Path) -> None:
        return None

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
        return None

    def link_input(self, path: Path) -> Optional[ArtifactRef]:
        return None

    def register_transform(
        self, *, key: str, kind: str, source: str, description: str
    ) -> Optional[TransformRef]:
        return None

    def set_end_status(
        self,
        status: Optional[RunStatus],
        session_id: Optional[str] = None,
        reason: Optional[str] = None,
        *,
        if_unset: bool = False,
    ) -> None:
        return None

    def end_run(
        self,
        status: RunStatus,
        report_path: Optional[Path] = None,
        environment_path: Optional[Path] = None,
        reason: Optional[str] = None,
    ) -> None:
        return None

    def status(self) -> CatalogStatus:
        return CatalogStatus(engine="none", mode="off", reason=self.reason)
