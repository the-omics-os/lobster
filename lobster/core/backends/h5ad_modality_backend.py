"""H5ADModalityBackend — WorkspaceModalityBackend backed by H5AD files.

First concrete implementation of WorkspaceModalityBackend. Wraps existing
H5AD file storage with lazy metadata access (h5py header reads), backed-mode
loading for large files, and round-trip AnnData persistence via H5ADBackend.

Architecture:
    workspace/
      data/
        dataset_a.h5ad
        dataset_b.h5ad
        ...

Each modality = one .h5ad file in the workspace modalities directory.
Metadata is read from HDF5 headers without loading X into memory.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np
import pandas as pd
from anndata import AnnData

from lobster.core.backends.modality_backend import (
    ModalityMetadata,
    ModalityRecord,
    ModalitySpec,
    WorkspaceModalityBackend,
)

logger = logging.getLogger(__name__)


def _peek_h5ad_metadata(path: Path) -> Dict[str, Any]:
    """Read shape + column schemas from h5ad header without loading X.

    ~1ms for any file size. Uses h5py direct access to HDF5 groups.
    """
    meta: Dict[str, Any] = {"path": path}
    try:
        with h5py.File(path, "r") as f:
            if "X" in f:
                x = f["X"]
                if isinstance(x, h5py.Dataset):
                    meta["n_obs"], meta["n_vars"] = int(x.shape[0]), int(x.shape[1])
                elif isinstance(x, h5py.Group):
                    shape = x.attrs.get("shape", None)
                    if shape is not None:
                        meta["n_obs"] = int(shape[0])
                        meta["n_vars"] = int(shape[1])
                    else:
                        meta["n_obs"] = int(f["obs"].attrs.get("_index_length", 0))
                        meta["n_vars"] = int(f["var"].attrs.get("_index_length", 0))

            if "n_obs" not in meta:
                meta["n_obs"] = int(f["obs"].attrs.get("_index_length", 0))
            if "n_vars" not in meta:
                meta["n_vars"] = int(f["var"].attrs.get("_index_length", 0))

            obs_cols = {}
            if "obs" in f:
                for key in f["obs"].keys():
                    if key.startswith("__"):
                        continue
                    ds = f["obs"][key]
                    dtype_str = str(ds.dtype) if hasattr(ds, "dtype") else "object"
                    if hasattr(ds, "attrs") and "categories" in ds.attrs:
                        dtype_str = "category"
                    obs_cols[key] = dtype_str
            meta["obs_columns"] = obs_cols

            var_cols = {}
            if "var" in f:
                for key in f["var"].keys():
                    if key.startswith("__"):
                        continue
                    ds = f["var"][key]
                    dtype_str = str(ds.dtype) if hasattr(ds, "dtype") else "object"
                    if hasattr(ds, "attrs") and "categories" in ds.attrs:
                        dtype_str = "category"
                    var_cols[key] = dtype_str
            meta["var_columns"] = var_cols

            meta["layers"] = list(f["layers"].keys()) if "layers" in f else []
            meta["obsm_keys"] = list(f["obsm"].keys()) if "obsm" in f else []
            meta["uns_keys"] = list(f["uns"].keys()) if "uns" in f else []

    except Exception as e:
        logger.warning(f"Failed to peek h5ad metadata from {path}: {e}")
        meta.setdefault("n_obs", 0)
        meta.setdefault("n_vars", 0)

    return meta


class H5ADModalityBackend(WorkspaceModalityBackend):
    """WorkspaceModalityBackend implementation using H5AD file storage.

    Each modality is a single .h5ad file in ``data_dir`` (typically
    ``workspace/data/``). Metadata is read from HDF5 headers (~1ms).
    Large files (>1GB) use backed mode.
    """

    def __init__(
        self,
        data_dir: Path,
        backed_mode: bool = False,
    ):
        self._data_dir = Path(data_dir)
        self._backed_mode = backed_mode
        self._dirty: set = set()

    def _h5ad_path(self, name: str) -> Path:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid modality name: {name!r}")
        path = (self._data_dir / f"{name}.h5ad").resolve()
        if path.parent != self._data_dir.resolve():
            raise ValueError(f"Invalid modality path: {name!r}")
        return path

    def _scan_files(self) -> Dict[str, Path]:
        """Return dict of modality_name -> h5ad path for all files."""
        if not self._data_dir.exists():
            return {}
        return {p.stem: p for p in sorted(self._data_dir.glob("*.h5ad"))}

    def list_modalities(self) -> List[ModalityRecord]:
        records = []
        for name, path in self._scan_files().items():
            try:
                meta = _peek_h5ad_metadata(path)
                records.append(
                    ModalityRecord(
                        name=name,
                        n_obs=meta.get("n_obs", 0),
                        n_vars=meta.get("n_vars", 0),
                        is_backed=False,
                        storage_size_bytes=path.stat().st_size,
                    )
                )
            except Exception as e:
                logger.warning(f"Skipping modality '{name}': {e}")
        return records

    def get_metadata(self, name: str) -> ModalityMetadata:
        path = self._h5ad_path(name)
        if not path.exists():
            raise KeyError(f"Modality '{name}' not found")

        meta = _peek_h5ad_metadata(path)
        return ModalityMetadata(
            name=name,
            n_obs=meta.get("n_obs", 0),
            n_vars=meta.get("n_vars", 0),
            obs_columns=meta.get("obs_columns", {}),
            var_columns=meta.get("var_columns", {}),
            layers=meta.get("layers", []),
            obsm_keys=meta.get("obsm_keys", []),
            uns_keys=meta.get("uns_keys", []),
            storage_size_bytes=path.stat().st_size,
            data_status="cold",
        )

    def query_obs(
        self,
        name: str,
        columns: Optional[Sequence[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> pd.DataFrame:
        path = self._h5ad_path(name)
        if not path.exists():
            raise KeyError(f"Modality '{name}' not found")

        import anndata

        adata = anndata.read_h5ad(path, backed="r")
        try:
            obs = adata.obs
            if filters:
                for col, val in filters.items():
                    if col in obs.columns:
                        obs = obs[obs[col] == val]
            if columns:
                available = [c for c in columns if c in obs.columns]
                obs = obs[available]
            obs = obs.iloc[offset:]
            if limit is not None:
                obs = obs.iloc[:limit]
            return obs.copy()
        finally:
            if hasattr(adata, "file"):
                try:
                    adata.file.close()
                except Exception:
                    pass

    def materialize_anndata(
        self,
        name: str,
        obs_filter: Optional[Dict[str, Any]] = None,
        features: Optional[Sequence[str]] = None,
    ) -> AnnData:
        path = self._h5ad_path(name)
        if not path.exists():
            raise KeyError(f"Modality '{name}' not found")

        from lobster.core.runtime.data_manager import (
            _close_backed_handle,
            _read_h5ad_smart,
        )

        adata = _read_h5ad_smart(path, backed_mode_enabled=self._backed_mode)

        if getattr(adata, "isbacked", False):
            backed = adata
            try:
                adata = backed.to_memory()
            finally:
                _close_backed_handle(backed)

        if obs_filter:
            mask = pd.Series(True, index=adata.obs.index)
            for col, val in obs_filter.items():
                if col in adata.obs.columns:
                    mask &= adata.obs[col] == val
            adata = adata[mask].copy()

        if features:
            valid = [f for f in features if f in adata.var_names]
            if valid:
                adata = adata[:, valid].copy()

        return adata

    def ingest_anndata(
        self,
        name: str,
        adata: AnnData,
        spec: Optional[ModalitySpec] = None,
    ) -> None:
        import os
        import tempfile

        path = self._h5ad_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)

        from lobster.core.backends.h5ad_backend import H5ADBackend

        backend = H5ADBackend(base_path=self._data_dir)
        fd, tmp_path = tempfile.mkstemp(suffix=".h5ad.tmp", dir=path.parent)
        os.close(fd)
        try:
            backend.save(adata, Path(tmp_path))
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        logger.info(f"Ingested modality '{name}' ({adata.n_obs} obs x {adata.n_vars} vars)")

    def export_h5ad(self, name: str, target: Path) -> Path:
        path = self._h5ad_path(name)
        if not path.exists():
            raise KeyError(f"Modality '{name}' not found")

        import shutil

        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)

        if path == target.resolve():
            return target

        shutil.copy2(path, target)
        return target

    def remove(self, name: str) -> None:
        path = self._h5ad_path(name)
        if not path.exists():
            raise KeyError(f"Modality '{name}' not found")
        path.unlink()
        self._dirty.discard(name)
        logger.info(f"Removed modality '{name}' from disk")

    def exists(self, name: str) -> bool:
        return self._h5ad_path(name).exists()

    def mark_dirty(self, name: str) -> None:
        self._dirty.add(name)

    def flush(self) -> None:
        if self._dirty:
            logger.warning(
                f"flush() called with {len(self._dirty)} dirty modalities "
                f"({', '.join(sorted(self._dirty))}). "
                f"Caller must persist via ingest_anndata() before flushing."
            )
        self._dirty.clear()

    def close(self) -> None:
        if self._dirty:
            logger.warning(
                f"close() called with {len(self._dirty)} unpersisted dirty modalities"
            )
        self._dirty.clear()
