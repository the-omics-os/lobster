#!/usr/bin/env python3
"""Verify that every analysis path Lobster agents depend on actually runs.

Why this exists
---------------
A specialist agent that errors on a missing or misconfigured dependency does not
merely fail: the supervisor then writes ``execute_custom_code`` to work around it,
and sometimes reports success for work that never happened. Environment faults
therefore masquerade as routing and escalation problems. This script separates the
two by proving, on real (tiny) data, that each analysis path executes.

Design notes
------------
* Every check is independent and reported separately. One missing optional
  dependency must never mask the rest.
* A check whose package is legitimately absent reports SKIPPED, not FAILED, so
  this is runnable in a core-only install.
* Deterministic in-process data. No network, no credentials, no model calls, and
  nothing written into a user workspace.
* Exit code is 0 only when nothing FAILED. SKIPPED does not fail the run.

Usage
-----
    python scripts/verify_env.py
    python scripts/verify_env.py --strict   # SKIPPED also fails (full installs)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import traceback
import warnings
from collections.abc import Callable

warnings.filterwarnings("ignore")

PASSED, SKIPPED, FAILED = "PASSED", "SKIPPED", "FAILED"


class Skip(Exception):
    """Raised by a check when a legitimately optional dependency is absent."""


def _require(*modules: str) -> None:
    """Skip the check unless every named module is importable."""
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        raise Skip(f"not installed: {', '.join(missing)}")


def _tiny_adata(n_obs: int = 60, n_vars: int = 40):
    """Build a small deterministic AnnData for the scanpy paths."""
    import anndata as ad
    import numpy as np

    rng = np.random.default_rng(0)
    counts = rng.poisson(1.0, (n_obs, n_vars)).astype("float32")
    return ad.AnnData(counts)


def _normalized_adata():
    import scanpy as sc

    adata = _tiny_adata()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def c_numba_threading() -> str:
    """Report the numba threading layer actually in use.

    Numba prefers tbb > omp > workqueue. ``tbb`` is fork-safe and composes under
    nested parallelism, which matters because Lobster runs tool calls
    concurrently via ``asyncio.to_thread``. But Intel publishes no ``tbb`` wheel
    for arm64, and both dev macOS and production ECS (Graviton) are arm64 — so
    ``workqueue`` is the only layer available there.

    We therefore verify and report rather than force. Forcing
    NUMBA_THREADING_LAYER=tbb where the library is absent turns a working run
    into a hard ValueError, which is strictly worse than the fallback.
    """
    _require("numba")
    import numba
    import numpy as np
    from numba import njit, prange

    @njit(parallel=True, cache=False)
    def _sum(values):
        total = 0.0
        for i in prange(values.size):
            total += values[i]
        return total

    _sum(np.arange(64.0))
    layer = numba.threading_layer()
    requested = os.environ.get("NUMBA_THREADING_LAYER")

    detail = f"layer in use: {layer}"
    if requested:
        detail += f" (NUMBA_THREADING_LAYER={requested})"
    if layer == "workqueue":
        detail += " — single-level fallback; see docstring"
    return detail


def c_pca() -> str:
    _require("scanpy", "anndata", "numpy")
    import scanpy as sc

    adata = _normalized_adata()
    sc.pp.pca(adata, n_comps=5)
    return f"X_pca {adata.obsm['X_pca'].shape}"


def c_neighbors() -> str:
    _require("scanpy", "anndata")
    import scanpy as sc

    adata = _normalized_adata()
    sc.pp.pca(adata, n_comps=5)
    sc.pp.neighbors(adata, n_neighbors=5)
    return "neighbors graph built"


def c_leiden() -> str:
    """Leiden clustering. leidenalg/igraph are C++, but the preceding
    neighbor-graph step is numba-backed, which is where threading matters."""
    _require("scanpy", "leidenalg", "igraph")
    import scanpy as sc

    adata = _normalized_adata()
    sc.pp.pca(adata, n_comps=5)
    sc.pp.neighbors(adata, n_neighbors=5)
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False)
    return f"{adata.obs['leiden'].nunique()} clusters"


def c_umap() -> str:
    """UMAP is the heaviest numba consumer in the stack."""
    _require("scanpy", "umap")
    import scanpy as sc

    adata = _normalized_adata()
    sc.pp.pca(adata, n_comps=5)
    sc.pp.neighbors(adata, n_neighbors=5)
    sc.tl.umap(adata)
    return f"X_umap {adata.obsm['X_umap'].shape}"


def c_scikit_survival() -> str:
    """scikit-survival backs survival_analysis_expert.

    This dependency was previously extras-only in lobster-ml, so the agent
    shipped and advertised survival tools that failed at call time. This dependency
    is required by lobster-ml because the agent ships with those tools.
    """
    _require("sksurv", "numpy")
    import numpy as np
    from sksurv.linear_model import CoxPHSurvivalAnalysis

    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    y = np.array(
        [(bool(i % 2), float(5 + i)) for i in range(40)],
        dtype=[("event", "?"), ("time", "<f8")],
    )
    CoxPHSurvivalAnalysis().fit(x, y)
    return "CoxPHSurvivalAnalysis fit"


def c_lifelines() -> str:
    """lifelines backs the proteomics survival path."""
    _require("lifelines")
    from lifelines import KaplanMeierFitter

    fitter = KaplanMeierFitter()
    fitter.fit([5, 6, 6, 2, 4, 4], [1, 0, 1, 1, 1, 0])
    return "KaplanMeierFitter fit"


def c_kaleido() -> str:
    """Static PNG export. Needs kaleido AND a working Chrome/Chromium."""
    _require("plotly", "kaleido")
    import plotly.graph_objects as go

    figure = go.Figure(go.Scatter(x=[1, 2], y=[1, 2]))
    payload = figure.to_image(format="png")
    if not payload:
        raise RuntimeError("kaleido returned an empty image")
    return f"PNG export {len(payload)} bytes"


CHECKS: list[tuple[str, Callable[[], str]]] = [
    ("numba threading layer", c_numba_threading),
    ("scanpy PCA", c_pca),
    ("scanpy neighbors", c_neighbors),
    ("leiden clustering", c_leiden),
    ("UMAP", c_umap),
    ("scikit-survival", c_scikit_survival),
    ("lifelines", c_lifelines),
    ("kaleido PNG export", c_kaleido),
]


def run_check(name: str, fn: Callable[[], str], verbose: bool) -> tuple[str, str]:
    """Run one check, converting any outcome into a (status, detail) pair."""
    try:
        return PASSED, fn()
    except Skip as exc:
        return SKIPPED, str(exc)
    except BaseException as exc:  # noqa: BLE001 - a check must never propagate
        if verbose:
            traceback.print_exc()
        return FAILED, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat SKIPPED as failure (use on full installs and release gates)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print tracebacks for failures"
    )
    args = parser.parse_args()

    print("Lobster environment verification")
    print(f"python {sys.version.split()[0]} on {sys.platform}")
    print("-" * 68)

    results = []
    for name, fn in CHECKS:
        status, detail = run_check(name, fn, args.verbose)
        results.append((name, status, detail))
        print(f"{status:<8} {name:<24} {detail}")

    failed = [r for r in results if r[1] == FAILED]
    skipped = [r for r in results if r[1] == SKIPPED]
    print("-" * 68)
    print(
        f"{len(results) - len(failed) - len(skipped)} passed, "
        f"{len(failed)} failed, {len(skipped)} skipped"
    )

    if failed:
        print("\nFAILED:")
        for name, _, detail in failed:
            print(f"  - {name}: {detail}")
        return 1

    if skipped and args.strict:
        print("\n--strict: skipped checks are failures here.")
        for name, _, detail in skipped:
            print(f"  - {name}: {detail}")
        return 1

    print("\nSUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
