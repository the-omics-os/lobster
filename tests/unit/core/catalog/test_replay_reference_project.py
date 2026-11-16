"""The replay reference in a project's catalog, and across a session restored later.

A project's catalog keys a file from the project folder, and a session restored in
another process records a Run of its own under the same name. Runs in its own
Python process, like ``test_lamin_project_runs``. Runs only with the ``lamin`` extra.
"""

import importlib.util
import textwrap

import pytest

from .test_lamin_engine import _run
from .test_lamin_project_runs import project, run  # noqa: F401  (project is a fixture)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        importlib.util.find_spec("lamindb_setup") is None
        or importlib.util.find_spec("lamindb") is None,
        reason="needs the lamin extra (lamindb-core)",
    ),
]

SCENARIO = """
    import hashlib
    from pathlib import Path
    import anndata as ad, numpy as np
    from lobster.core.catalog import factory
    from lobster.core.runtime.data_manager import DataManagerV2
    from lobster.cli_internal.commands.light import lamin_commands

    proj = Path(os.environ["PROJ"])
    ws = proj / ".lobster_workspace"
    NB = '{"cells": [], "metadata": {}}'

    def session(sid, saves, notebook=False):
        dm = DataManagerV2(workspace_path=ws)
        catalog = factory.begin_session_run(dm, ws, sid, ws / ".lobster" / "sessions" / sid)
        for stem, seed in saves:
            rng = np.random.default_rng(seed)
            a = ad.AnnData(rng.poisson(1.0, size=(30, 8)).astype(np.float32))
            a.obs_names = [f"c{i}" for i in range(30)]
            a.var_names = [f"g{i}" for i in range(8)]
            dm.store_modality(stem, a)
            dm.save_modality(stem, f"{stem}.h5ad")
        if notebook:
            catalog.register_transform(key="notebooks/n.ipynb", kind="notebook",
                                       source=NB, description="the export")
        catalog.close("completed")

    step = os.environ["STEP"]
    if step == "one":
        session("session_one", [("x", 1)])
    elif step == "other":
        session("session_other", [("y", 2)])
    elif step == "restore":  # the same session, continued in another process
        session("session_one", [], notebook=True)
    if step != "replay":
        emit(done=step)
        raise SystemExit(0)
    eng = lamin_commands._open(ws)
    started = eng.begin_replay(key="notebooks/n.ipynb", source=NB, params={},
                               lobster_version="0.0.0")
    x, y = eng.replay_reference_state("x"), eng.replay_reference_state("y")
    emit(session=started.session_uid is not None,
         x=[None if x[0] is None else str(x[0].relative_to(proj)), x[1]],
         y=[None if y[0] is None else str(y[0]), y[1]])
"""


def _scenario(tmp_path, project):  # noqa: F811
    for step in ("one", "other", "restore"):
        run(SCENARIO, tmp_path, project, STEP=step)
    return run(SCENARIO, tmp_path, project, STEP="replay")


def test_the_reference_is_found_in_a_project_across_a_restored_session(
    tmp_path, project  # noqa: F811
):
    r = _scenario(tmp_path, project)
    assert r["session"]
    assert r["x"] == [".lobster_workspace/data/x.h5ad", "ok"], r


def test_a_run_of_another_session_is_never_the_reference(
    tmp_path, project
):  # noqa: F811
    r = _scenario(tmp_path, project)
    assert r["y"] == [None, "none"], r
