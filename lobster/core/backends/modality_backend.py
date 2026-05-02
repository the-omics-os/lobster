"""WorkspaceModalityBackend — abstract interface for modality storage backends.

Owns the contract that all modality backends (H5AD, Zarr, future homeobox)
must satisfy. Defined by Omics-OS, not by any external dependency.

This is the Phase 2 replacement for the raw ``Dict[str, AnnData]`` in
DataManagerV2. Each backend implementation provides lazy metadata access,
filtered materialization, and round-trip AnnData persistence.

Architecture:
    Agent -> DataManagerV2 -> WorkspaceModalityBackend -> storage
    get_modality() returns concrete AnnData (via materialize_anndata).
    The 22-agent contract is SACRED: agents always receive real AnnData.

See Also:
    - lobster/core/interfaces/backend.py — IDataBackend (file-level I/O)
    - lobster/core/runtime/data_manager.py — DataManagerV2 (orchestration)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from anndata import AnnData
from pandas import DataFrame


@dataclass(frozen=True)
class ModalityRecord:
    """Lightweight record returned by list_modalities().

    Contains only what's needed for UI rendering and routing —
    no heavy data. Cheap to compute from metadata alone.
    """

    name: str
    n_obs: int
    n_vars: int
    omics_type: str = "unknown"
    is_backed: bool = False
    created_at: Optional[datetime] = None
    parent_modality: Optional[str] = None
    processing_step: str = "raw"
    storage_size_bytes: Optional[int] = None


@dataclass
class ModalityMetadata:
    """Rich metadata for a single modality.

    Returned by get_metadata(). Includes column schemas for obs/var,
    available layers, and storage details. Sufficient for the cloud
    metadata-only serving path (no AnnData materialization needed).
    """

    name: str
    n_obs: int
    n_vars: int
    omics_type: str = "unknown"
    obs_columns: Dict[str, str] = field(default_factory=dict)
    var_columns: Dict[str, str] = field(default_factory=dict)
    layers: List[str] = field(default_factory=list)
    obsm_keys: List[str] = field(default_factory=list)
    uns_keys: List[str] = field(default_factory=list)
    is_backed: bool = False
    storage_size_bytes: Optional[int] = None
    created_at: Optional[datetime] = None
    data_status: str = "cold"


@dataclass(frozen=True)
class ModalitySpec:
    """Ingestion specification for a modality.

    Tells the backend how to store a new AnnData — which field
    holds the primary expression matrix, which omics type it is,
    and any backend-specific hints.
    """

    omics_type: str = "unknown"
    field_name: str = "X"
    sparse: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


class WorkspaceModalityBackend(ABC):
    """Abstract interface for workspace-level modality storage.

    Unlike IDataBackend (file-level load/save), this interface operates
    at the modality level: list, query, materialize, ingest. Backends
    may use H5AD files, Zarr stores, or future homeobox atlases
    underneath — agents never know the difference.

    Contract guarantees:
        - materialize_anndata() ALWAYS returns a concrete, in-memory AnnData.
        - Returned AnnData passes isinstance(result, AnnData).
        - Scanpy pp/tl functions work on the returned AnnData without guards.
        - ingest_anndata() + materialize_anndata() is a lossless round-trip
          for .X, .obs, .var, .obsm, .layers, .uns (modulo float precision).
    """

    @abstractmethod
    def list_modalities(self) -> List[ModalityRecord]:
        """List all modalities in the workspace.

        Returns lightweight records — no data loaded. Must be fast
        (< 100ms) for UI rendering.
        """

    @abstractmethod
    def get_metadata(self, name: str) -> ModalityMetadata:
        """Get rich metadata for a modality without loading data.

        Includes column schemas (obs/var column names + dtypes),
        available layers, obsm keys. Sufficient for cloud
        metadata-only serving.

        Raises:
            KeyError: If modality not found.
        """

    @abstractmethod
    def query_obs(
        self,
        name: str,
        columns: Optional[Sequence[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> DataFrame:
        """Query observation metadata without materializing .X.

        Supports column selection, simple equality filters, and
        pagination. Returns a pandas DataFrame.

        This enables data preview, column inspection, and filter
        building without loading the full matrix.

        Args:
            name: Modality name.
            columns: Subset of obs columns to return (None = all).
            filters: Dict of column -> value equality filters.
            limit: Max rows to return (None = all).
            offset: Row offset for pagination.

        Raises:
            KeyError: If modality not found.
        """

    @abstractmethod
    def materialize_anndata(
        self,
        name: str,
        obs_filter: Optional[Dict[str, Any]] = None,
        features: Optional[Sequence[str]] = None,
    ) -> AnnData:
        """Materialize a modality as a concrete, in-memory AnnData.

        This is the ONLY way agents get AnnData from the backend.
        The returned object is fully in-memory — .X is a numpy/scipy
        array, not an HDF5 dataset or lazy proxy.

        Optional subsetting:
            obs_filter: subset observations (e.g., {"cell_type": "T cell"})
            features: subset to specific var_names

        Args:
            name: Modality name.
            obs_filter: Optional observation filters for subsetting.
            features: Optional feature names for column subsetting.

        Returns:
            Concrete AnnData fully loaded in memory.

        Raises:
            KeyError: If modality not found.
            MemoryError: If materialization exceeds available memory.
        """

    @abstractmethod
    def ingest_anndata(
        self,
        name: str,
        adata: AnnData,
        spec: Optional[ModalitySpec] = None,
    ) -> None:
        """Ingest an AnnData into the backend under the given name.

        If a modality with this name already exists, it is overwritten.
        The backend is responsible for serialization format and layout.

        Args:
            name: Modality name.
            adata: AnnData to store.
            spec: Optional ingestion specification (omics type, field hints).

        Raises:
            ValueError: If adata is invalid or cannot be serialized.
        """

    @abstractmethod
    def export_h5ad(self, name: str, target: Path) -> Path:
        """Export a modality to an H5AD file on disk.

        Used by custom_code_tool for subprocess serialization and
        by workspace export/download features.

        Args:
            name: Modality name.
            target: Destination file path.

        Returns:
            Path to the written H5AD file.

        Raises:
            KeyError: If modality not found.
        """

    @abstractmethod
    def remove(self, name: str) -> None:
        """Remove a modality from the backend.

        Frees storage and any open file handles.

        Raises:
            KeyError: If modality not found.
        """

    @abstractmethod
    def exists(self, name: str) -> bool:
        """Check if a modality exists in the backend."""

    def mark_dirty(self, name: str) -> None:
        """Mark a modality as modified (needs re-persist on save).

        Default implementation is a no-op — backends that track
        dirty state should override.
        """

    def flush(self) -> None:
        """Persist all dirty modalities to durable storage.

        Default implementation is a no-op — backends with deferred
        writes should override.
        """

    def close(self) -> None:
        """Release all resources (file handles, connections).

        Called on session teardown. Default is a no-op.
        """
