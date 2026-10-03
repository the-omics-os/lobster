"""Resolution of the ``.obs`` column holding cluster assignments.

The clustering service names columns by resolution (``leiden_res0_5``,
``louvain_res1_0``, ``leiden_subcluster``), while downstream readers may receive
the bare string ``"leiden"``. Readers must resolve a column actually present in
the data rather than assume the writer created a particular alias.

The resolution order is deliberate:

1. An explicit ``cluster_key`` always wins — the caller knows best, and a wrong
   explicit key must fail loudly rather than be silently replaced.
2. An exact ``leiden`` / ``louvain`` column. ``cluster_and_visualize()`` writes
   this as an alias for the primary resolution, so it is the common case.
3. The *unique* ``leiden_res*`` / ``louvain_res*`` column. If there is exactly
   one, there is no ambiguity to report.
4. Only then raise — and only when the choice is genuinely ambiguous or there is
   nothing to choose.

This module lives in core because the writer is in core
(``lobster/services/analysis/clustering_service.py``) and the readers are in the
``lobster-transcriptomics`` package; both sides must agree on the answer.
"""

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    import anndata

#: Clustering algorithms whose column names this module understands.
CLUSTER_ALGORITHMS = ("leiden", "louvain")


class ClusterKeyError(ValueError):
    """Raised when the cluster column cannot be determined unambiguously.

    Subclasses ``ValueError`` so existing ``except ValueError`` handlers around
    annotation keep working.
    """


def list_cluster_key_candidates(adata: "anndata.AnnData") -> List[str]:
    """All ``.obs`` columns that look like cluster assignments, sorted."""
    columns = list(getattr(adata, "obs", {}).columns)
    candidates = [
        col
        for col in columns
        if any(col == algo or col.startswith(f"{algo}_") for algo in CLUSTER_ALGORITHMS)
    ]
    return sorted(candidates)


def resolve_cluster_key(
    adata: "anndata.AnnData",
    cluster_key: Optional[str] = None,
    algorithm: Optional[str] = None,
) -> str:
    """Return the ``.obs`` column holding cluster assignments.

    Args:
        adata: Object whose ``.obs`` is searched.
        cluster_key: Explicit column name.  If given it is validated and
            returned unchanged — never silently substituted.
        algorithm: Restrict resolution to one algorithm (``"leiden"`` or
            ``"louvain"``).  Use this when the caller knows which algorithm ran;
            it is what makes a workspace holding both unambiguous.

    Returns:
        The resolved column name.

    Raises:
        ClusterKeyError: If an explicit ``cluster_key`` is absent, if no
            candidate column exists, or if the choice is genuinely ambiguous.
    """
    obs = getattr(adata, "obs", None)
    columns = list(obs.columns) if obs is not None else []

    # 1. Explicit wins, and a bad explicit key fails loudly.
    if cluster_key is not None:
        if cluster_key in columns:
            return cluster_key
        candidates = list_cluster_key_candidates(adata)
        hint = (
            f" Available cluster columns: {candidates}."
            if candidates
            else " No cluster columns found — run clustering first."
        )
        # Wording is "Cluster key", not "Cluster column": it is the pinned
        # public contract of compute_clustering_quality() and is asserted by
        # tests/unit/tools/test_clustering_quality.py.
        raise ClusterKeyError(
            f"Cluster key '{cluster_key}' not found in adata.obs.{hint}"
        )

    algorithms = (algorithm,) if algorithm in CLUSTER_ALGORITHMS else CLUSTER_ALGORITHMS

    # 2. Exact algorithm-named column (the alias clustering writes).
    exact = [algo for algo in algorithms if algo in columns]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ClusterKeyError(
            f"Ambiguous cluster column: adata.obs holds {exact}. "
            f"Pass cluster_key= explicitly, or algorithm= to choose."
        )

    # 3. The unique resolution-suffixed column.
    suffixed = [
        col
        for col in sorted(columns)
        for algo in algorithms
        if col.startswith(f"{algo}_")
    ]
    if len(suffixed) == 1:
        return suffixed[0]

    # 4. Genuinely ambiguous, or nothing to choose.
    if not suffixed:
        candidates = list_cluster_key_candidates(adata)
        detail = (
            f" Columns that look related: {candidates}."
            if candidates
            else " Run clustering before annotation."
        )
        raise ClusterKeyError(
            "No cluster assignments found in adata.obs (looked for "
            f"{', '.join(algorithms)} and their *_res* variants).{detail}"
        )

    raise ClusterKeyError(
        f"Ambiguous cluster column: {len(suffixed)} candidates in adata.obs "
        f"{suffixed}. Pass cluster_key= explicitly to choose one."
    )
