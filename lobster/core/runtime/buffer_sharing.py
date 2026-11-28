"""Share byte-identical array buffers between stored modalities.

A pipeline stores one AnnData per step, and consecutive steps often carry the
same matrix forward unchanged: quality assessment adds ``.obs`` columns but
keeps ``.X``; clustering keeps ``.X`` and snapshots it as ``.raw``; marker
ranking returns the clustered matrix again. Without sharing, every version owns
its own copy.

When a modality is stored, each large numpy buffer it holds (dense arrays and
the ``data``/``indices``/``indptr`` of CSR/CSC matrices) is hashed. If a
modality that is still resident already owns a buffer with the same dtype,
shape and bytes, the new object is pointed at that buffer and its duplicate is
released. Values are never changed, only the ownership of identical bytes.

Every buffer that ends up shared, and every buffer one object exposes in two
places (``.X`` and ``.raw.X``), is made read-only. An in-place write to a
shared buffer then raises ``ValueError`` instead of silently changing another
stored version. Buffers that are not shared stay writeable, so code that edits
a stored modality it alone owns is unaffected.

Set ``LOBSTER_SHARE_MODALITY_BUFFERS=0`` to disable sharing.
"""

from __future__ import annotations

import hashlib
import os
import threading
import weakref
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Optional, Tuple

import numpy as np
from scipy import sparse

ENV_VAR = "LOBSTER_SHARE_MODALITY_BUFFERS"
# Smaller buffers are not worth a hash; obs/var-sized arrays fall below this.
MIN_SHARED_BYTES = 1 << 20
_SPARSE_PARTS = ("data", "indices", "indptr")


def sharing_enabled() -> bool:
    """Sharing is on unless ``LOBSTER_SHARE_MODALITY_BUFFERS`` is 0/false/off."""
    value = os.environ.get(ENV_VAR, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def is_read_only(matrix: Any) -> bool:
    """True when every buffer of ``matrix`` is read-only (registry-frozen)."""
    if sparse.issparse(matrix):
        parts = [getattr(matrix, p, None) for p in _SPARSE_PARTS]
        parts = [p for p in parts if isinstance(p, np.ndarray)]
        return bool(parts) and not any(p.flags.writeable for p in parts)
    return isinstance(matrix, np.ndarray) and not matrix.flags.writeable


def has_no_stored_zeros(matrix: Any) -> bool:
    """Canonical CSR/CSC with no explicit zeros and no slack after ``nnz``.

    For such a matrix ``eliminate_zeros()`` removes nothing.
    """
    return (
        sparse.issparse(matrix)
        and matrix.format in ("csr", "csc")
        and matrix.has_canonical_format
        and matrix.indptr[-1] == matrix.data.size == matrix.indices.size
        and np.count_nonzero(matrix.data) == matrix.data.size
    )


_ELIMINATE_LOCK = threading.RLock()
_COMPRESSED_CLASSES = (
    sparse.csr_matrix,
    sparse.csc_matrix,
    sparse.csr_array,
    sparse.csc_array,
)


@contextmanager
def skip_noop_eliminate_zeros(*targets: Any):
    """Let shared, read-only matrices pass through Scanpy's ``eliminate_zeros()``.

    ``sc.tl.rank_genes_groups`` and ``sc.pp.calculate_qc_metrics`` call
    ``X.eliminate_zeros()`` on the matrix they are given. With no stored zeros
    that removes nothing, yet SciPy still rewrites every buffer, which a
    read-only shared buffer refuses. While active, the call returns early for
    matrices that share memory with ``targets`` *and* have nothing to remove.
    Every other matrix, including a target that does hold stored zeros (which
    then fails loudly on its read-only buffer), runs SciPy's method unchanged.
    """
    target_parts = [
        getattr(t, p)
        for t in targets
        if sparse.issparse(t)
        for p in _SPARSE_PARTS
        if isinstance(getattr(t, p, None), np.ndarray)
    ]
    if not target_parts:
        yield
        return

    def guarded(original):
        def eliminate_zeros(self):
            parts = [getattr(self, p, None) for p in _SPARSE_PARTS]
            if any(
                isinstance(a, np.ndarray) and np.shares_memory(a, b)
                for a in parts
                for b in target_parts
            ) and has_no_stored_zeros(self):
                return None
            return original(self)

        return eliminate_zeros

    with _ELIMINATE_LOCK:
        saved = []
        for cls in _COMPRESSED_CLASSES:
            own = cls.__dict__.get("eliminate_zeros")
            saved.append((cls, own))
            cls.eliminate_zeros = guarded(cls.eliminate_zeros)
        try:
            yield
        finally:
            for cls, own in saved:
                if own is None:
                    del cls.eliminate_zeros
                else:
                    cls.eliminate_zeros = own


def _iter_matrices(adata: Any) -> Iterator[Tuple[str, Any, Any, Any]]:
    """Yield (location, matrix, container, key) for every matrix-scale slot.

    ``container``/``key`` say how to rebind a dense array: ``container[key] = new``
    for mappings, ``setattr(adata, "X", new)`` when container is the AnnData.
    """
    # Backed .X lives on disk; annotations are still in memory.
    if not getattr(adata, "isbacked", False) and getattr(adata, "X", None) is not None:
        yield "X", adata.X, adata, "X"
    raw = getattr(adata, "raw", None)
    if raw is not None and getattr(raw, "X", None) is not None:
        yield "raw.X", raw.X, raw, "X"
    for attr in ("layers", "obsm", "varm", "obsp", "varp"):
        mapping = getattr(adata, attr, None)
        if mapping is None:
            continue
        try:
            keys = list(mapping.keys())
        except Exception:  # nosec B112 - unreadable optional mapping (e.g. a mock)
            continue
        for key in keys:
            # anndata >= 0.13 exposes .X as layers[None]; it is handled above.
            if attr == "layers" and key is None:
                continue
            yield f"{attr}[{key!r}]", mapping[key], mapping, key


def _iter_buffers(adata: Any) -> Iterator[np.ndarray]:
    for _, matrix, _, _ in _iter_matrices(adata):
        if sparse.issparse(matrix):
            # CSR/CSC: data/indices/indptr; COO: data/row/col.
            for part in ("data", "indices", "indptr", "row", "col"):
                buffer = getattr(matrix, part, None)
                if isinstance(buffer, np.ndarray):
                    yield buffer
        elif isinstance(matrix, np.ndarray):
            yield matrix


def unique_resident_bytes(adatas: Iterable[Any]) -> int:
    """Bytes held by the given AnnData objects, counting a shared buffer once.

    Mirrors ``estimate_anndata_bytes`` (matrix-scale containers only) but
    deduplicates by buffer identity, so sharing is reflected in the budget.
    """
    from lobster.core.runtime.data_manager import estimate_anndata_bytes

    seen = set()
    total = 0
    for adata in adatas:
        if adata is None:
            continue
        own = 0
        found = False
        for buffer in _iter_buffers(adata):
            found = True
            if id(buffer) in seen:
                continue
            seen.add(id(buffer))
            own += int(buffer.nbytes)
        # Objects without numpy buffers (mocks, backed or exotic arrays) fall
        # back to the per-object estimate.
        total += own if found else estimate_anndata_bytes(adata)
    return total


class BufferPool:
    """Content-addressed index of the buffers held by resident modalities."""

    def __init__(self) -> None:
        # (dtype, shape, digest) -> (array weakref, owning modality name)
        self._by_key: dict = {}
        # id(array) -> (array weakref, key, owning modality name)
        self._by_id: dict = {}
        self.shared_bytes = 0
        self.hashed_bytes = 0

    def _key(self, array: np.ndarray):
        cached = self._by_id.get(id(array))
        if cached is not None and cached[0]() is array:
            return cached[1], cached[2]
        contiguous = np.ascontiguousarray(array)
        digest = hashlib.blake2b(
            memoryview(contiguous).cast("B"), digest_size=32
        ).hexdigest()
        self.hashed_bytes += array.nbytes
        return (array.dtype.str, array.shape, digest), None

    def _resolve(
        self, array: Any, name: str, resident: set, seen_now: set
    ) -> Optional[np.ndarray]:
        """Return the buffer ``array`` should become (itself or a shared one)."""
        if (
            not isinstance(array, np.ndarray)
            or array.dtype.hasobject
            or array.nbytes < MIN_SHARED_BYTES
        ):
            return array
        key, owner = self._key(array)
        if id(array) in seen_now or (
            owner is not None and owner != name and owner in resident
        ):
            # The same object is reachable from two places: freeze it in place.
            array.flags.writeable = False
        seen_now.add(id(array))

        entry = self._by_key.get(key)
        existing = entry[0]() if entry is not None else None
        if (
            existing is not None
            and existing is not array
            and entry[1] in resident
            and entry[1] != name
        ):
            existing.flags.writeable = False
            self.shared_bytes += array.nbytes
            seen_now.add(id(existing))
            return existing

        if existing is None or entry[1] not in resident:
            self._by_key[key] = (weakref.ref(array), name)
        if owner is None or owner not in resident or owner == name:
            self._by_id[id(array)] = (weakref.ref(array), key, name)
        return array

    def share(self, name: str, adata: Any, resident: Iterable[str]) -> int:
        """Point ``adata`` at identical buffers already held by ``resident``.

        ``resident`` lists the other modalities still held in memory; buffers
        owned by anything else (spilled, removed, or the entry ``name`` itself
        is replacing) are never shared. Returns the bytes released.
        """
        resident = set(resident)
        resident.discard(name)
        before = self.shared_bytes
        seen_now: set = set()
        for _, matrix, container, key in list(_iter_matrices(adata)):
            if sparse.issparse(matrix) and matrix.format in ("csr", "csc"):
                for part in _SPARSE_PARTS:
                    current = getattr(matrix, part)
                    target = self._resolve(current, name, resident, seen_now)
                    if target is not current:
                        setattr(matrix, part, target)
                # A matrix is frozen as a whole: indptr is small enough to skip
                # sharing, but in-place writers (eliminate_zeros) rewrite it too.
                parts = [getattr(matrix, part) for part in _SPARSE_PARTS]
                if any(not p.flags.writeable for p in parts):
                    for p in parts:
                        p.flags.writeable = False
            elif isinstance(matrix, np.ndarray):
                target = self._resolve(matrix, name, resident, seen_now)
                if target is not matrix:
                    if container is adata and key == "X":
                        adata.X = target
                    elif (
                        key == "X"
                    ):  # dense .raw.X: rebuild Raw around the shared array
                        from anndata import AnnData

                        raw = adata.raw
                        adata.raw = AnnData(X=target, var=raw.var, varm=dict(raw.varm))
                    else:
                        container[key] = target
        self._prune()
        return self.shared_bytes - before

    def _prune(self) -> None:
        self._by_key = {k: v for k, v in self._by_key.items() if v[0]() is not None}
        self._by_id = {k: v for k, v in self._by_id.items() if v[0]() is not None}
