"""
Data locality provider interface.

Defines the contract for materializing modalities to local disk,
enabling subprocess-based execution (custom code, notebooks) to
access data regardless of where it actually resides (local, S3, GCS).

The engine defines the protocol; storage-specific implementations
(S3, API, etc.) are registered by the hosting environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lobster.core.data_manager_v2 import DataManagerV2


def _safe_filename(modality_name: str) -> str:
    """Sanitize modality name for use as a filename. Prevents path traversal."""
    safe = Path(modality_name).name.replace("..", "_")
    if not safe:
        safe = "_unnamed"
    return safe


@runtime_checkable
class DataLocalityProvider(Protocol):
    """Protocol for materializing modalities to local disk.

    The engine's materializer calls ensure_local() ONLY when:
    - The modality is NOT in DataManagerV2.modalities (not in memory), OR
    - The modality IS in memory but is backed (adata.isbacked == True)

    Implementations must either return a valid local Path to an .h5ad file
    or raise FileNotFoundError with a message that includes list_available().
    """

    def ensure_local(self, modality_name: str, target_dir: Path) -> Path:
        """Materialize a modality to local disk.

        Args:
            modality_name: Name of the modality to materialize.
            target_dir: Directory to write the .h5ad file into.

        Returns:
            Path to the local .h5ad file.

        Raises:
            FileNotFoundError: If the modality cannot be found anywhere.
        """
        ...

    def list_available(self) -> list[str]:
        """List ALL modalities that can be materialized.

        Returns a superset of DataManagerV2.list_modalities(), including
        remote/persisted modalities not currently loaded in memory.
        """
        ...

    def is_local(self, modality_name: str) -> bool:
        """Check if a modality already exists as a local file on disk."""
        ...


class LocalDataLocalityProvider:
    """Default provider for local CLI — files already on disk.

    This provider checks the data directory for existing .h5ad files.
    It is the no-op default: local CLI behavior is unchanged.
    """

    def __init__(self, data_manager: DataManagerV2) -> None:
        self._dm = data_manager

    def ensure_local(self, modality_name: str, target_dir: Path) -> Path:
        safe_name = _safe_filename(modality_name)
        data_path = self._dm.data_dir / f"{safe_name}.h5ad"
        if data_path.exists():
            return data_path
        available = self.list_available()
        raise FileNotFoundError(
            f"Modality '{modality_name}' not found on local disk. "
            f"Available: {available}"
        )

    def list_available(self) -> list[str]:
        names = set(self._dm.list_modalities())
        if self._dm.data_dir.exists():
            names |= {f.stem for f in self._dm.data_dir.glob("*.h5ad")}
        return sorted(names)

    def is_local(self, modality_name: str) -> bool:
        safe_name = _safe_filename(modality_name)
        return (self._dm.data_dir / f"{safe_name}.h5ad").exists()
