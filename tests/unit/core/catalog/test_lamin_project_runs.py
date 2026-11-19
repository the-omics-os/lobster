"""The LaminDB catalog of a project: every run is recorded in the project's catalog.

Each scenario runs in its own Python process (see ``test_lamin_engine``): LaminDB
keeps process-wide settings. Runs only with the ``lamin`` extra.
"""

import importlib.util
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from lobster.core.projects import init_project

from .test_lamin_engine import PRELUDE, _isolated_env, _run

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        importlib.util.find_spec("lamindb_setup") is None
        or importlib.util.find_spec("lamindb") is None,
        reason="needs the lamin extra (lamindb-core)",
    ),
]

RUNS = """
    import time
    from pathlib import Path
    import anndata as ad, numpy as np
    from lobster.core.catalog import factory
    from lobster.core.runtime.data_manager import DataManagerV2

    proj = Path(os.environ["PROJ"])
    seed = int(os.environ.get("SEED", "0"))
    go = os.environ.get("GO")
    while go and not Path(go).exists():
        time.sleep(0.02)
    def adata(n):
        rng = np.random.default_rng(n)
        a = ad.AnnData(rng.poisson(1.0, size=(30, 8)).astype(np.float32))
        a.obs_names = [f"c{i}" for i in range(30)]
        a.var_names = [f"g{i}" for i in range(8)]
        return a

    kinds, catalog = [], None
    for i, name in enumerate(os.environ["RUNS"].split(",")):
        ws = proj / "runs" / name / ".lobster_workspace"
        ws.mkdir(parents=True, exist_ok=True)
        dm = DataManagerV2(workspace_path=ws)
        sid = f"session_{name}"
        catalog = factory.begin_session_run(dm, ws, sid, ws / ".lobster" / "sessions" / sid)
        kinds.append(type(catalog).__name__)
        dm.store_modality("x", adata(seed + i))
        dm.save_modality("x", "x.h5ad")
        factory.note_run_status(dm, "completed")
    status = catalog.status()
    getattr(catalog, "close", lambda status: None)("completed")
    emit(kinds=kinds, degraded=status.degraded, reason=status.reason,
         location=status.location, instance=status.instance)
"""

READBACK = """
    import sqlite3
    from pathlib import Path
    proj = Path(os.environ["PROJ"])
    os.environ["LAMIN_SETTINGS_DIR"] = str(proj / ".lamindb" / "settings")
    import lamindb_setup as lns
    slug = json.loads((proj / ".lamindb" / "lobster.json").read_text())["slug"]
    lns.connect(slug, _write_settings=False)
    import lamindb as ln
    runs = list(ln.Run.filter(transform__key="lobster-ai"))
    arts = list(ln.Artifact.filter(key__startswith="runs/").order_by("key"))
    emit(slug=slug, runs=sorted(r.name for r in runs),
         statuses=sorted({r.status for r in runs}),
         keys=[a.key for a in arts], versions=[a.version for a in arts],
         latest=[a.is_latest for a in arts],
         in_place=[Path(str(a.path)).resolve() == (proj / a.key).resolve() for a in arts],
         outputs={r.name: sorted(a.key for a in r.output_artifacts.all()) for r in runs},
         storage=str(lns.settings.instance.storage.root),
         integrity=sqlite3.connect(proj / ".lamindb" / "lamin.db")
         .execute("PRAGMA integrity_check").fetchone()[0])
"""


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    init_project(root, "study")
    # The flag init --lamin writes; the shell has none.
    (root / ".lobster_workspace" / ".env").write_text("LOBSTER_LAMIN=1\n")
    return root.resolve()


def run(code, tmp_path, project, **env):
    env = {"PROJ": str(project), **env}
    return _run(code, tmp_path, env_extra=env, drop_enable=True)


def readback(tmp_path, project):
    return run(READBACK, tmp_path, project)


def test_every_run_in_one_server_is_recorded_in_the_projects_catalog(tmp_path, project):
    first = run(RUNS, tmp_path, project, RUNS="a,b,c")
    assert first["kinds"] == ["ThreadedCatalog"] * 3
    assert not first["degraded"], first["reason"]
    assert first["net"] == []
    assert Path(first["location"]) == project
    back = readback(tmp_path, project)
    assert back["runs"] == ["session_a", "session_b", "session_c"]
    assert back["statuses"] == ["completed"]
    assert back["keys"] == [
        f"runs/{n}/.lobster_workspace/data/x.h5ad" for n in ("a", "b", "c")
    ]
    assert Path(back["storage"]) == project


def test_two_runs_that_save_the_same_name_are_not_versions_of_one_file(
    tmp_path, project
):
    run(RUNS, tmp_path, project, RUNS="a,b")
    back = readback(tmp_path, project)
    assert len(back["keys"]) == 2 and len(set(back["keys"])) == 2
    assert back["versions"][0] == back["versions"][1]
    assert back["latest"] == [True, True]
    assert back["outputs"] == {
        "session_a": ["runs/a/.lobster_workspace/data/x.h5ad"],
        "session_b": ["runs/b/.lobster_workspace/data/x.h5ad"],
    }


def test_the_files_are_registered_where_they_are_and_the_catalog_is_in_one_place(
    tmp_path, project
):
    run(RUNS, tmp_path, project, RUNS="a,b")
    back = readback(tmp_path, project)
    assert back["in_place"] == [True, True]
    assert (
        sorted(p.name for p in project.glob("runs/*/.lobster_workspace/.lamindb")) == []
    )
    assert (project / ".lamindb" / "lobster.json").is_file()
    assert not (project / ".lobster_workspace" / ".lamindb").exists()


def test_a_later_server_adds_its_runs_to_the_same_catalog(tmp_path, project):
    run(RUNS, tmp_path, project, RUNS="a")
    run(RUNS, tmp_path, project, RUNS="b", SEED="5")
    back = readback(tmp_path, project)
    assert back["runs"] == ["session_a", "session_b"]
    assert len(back["keys"]) == 2


def test_servers_starting_together_on_one_project_all_record_without_corruption(
    tmp_path, project
):
    names = ["a", "b", "c", "d", "e", "f"]
    go = tmp_path / "go"
    procs = []
    for seed, name in enumerate(names):
        env = _isolated_env(
            tmp_path,
            {"PROJ": str(project), "RUNS": name, "SEED": str(seed), "GO": str(go)},
            True,
        )
        procs.append(
            subprocess.Popen(
                [sys.executable, "-c", PRELUDE + textwrap.dedent(RUNS)],
                cwd=tmp_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    time.sleep(4)
    go.write_text("go")
    results = []
    for proc in procs:
        out, err = proc.communicate(timeout=300)
        assert proc.returncode == 0, out[-2000:] + err[-2000:]
        line = [x for x in out.splitlines() if x.startswith("RESULT ")][-1]
        results.append(json.loads(line.removeprefix("RESULT ")))
    # The first start creates the catalog; the others wait for it and connect.
    assert [r["degraded"] for r in results] == [False] * len(names), results
    back = readback(tmp_path, project)
    assert back["keys"] == [f"runs/{n}/.lobster_workspace/data/x.h5ad" for n in names]
    assert back["runs"] == [f"session_{n}" for n in names]
    assert back["integrity"] == "ok"


# -- a catalog made before projects recorded their runs ---------------------------

OLD_CATALOG = """
    from pathlib import Path
    home = Path(os.environ["PROJ"]) / ".lobster_workspace"
    settings = home / ".lamindb" / "settings"
    settings.mkdir(parents=True)
    os.environ["LAMIN_SETTINGS_DIR"] = str(settings)
    import lamindb_setup as lns
    lns.settings.user
    lns.init(storage=str(home), name="lobster-old", _write_settings=False)
    iset = lns.settings.instance
    iset._persist(write_to_disk=True, write_current_instance_file=False)
    (home / ".lamindb" / "lobster.json").write_text(json.dumps({"slug": iset.slug}))
    emit(slug=iset.slug)
"""

OLD_USE = """
    from pathlib import Path
    from lobster.core.catalog.lamin import CatalogUnavailable, LaminCatalog
    proj = Path(os.environ["PROJ"])
    own = LaminCatalog(proj / ".lobster_workspace", create=False)
    root = str(own._storage_root)
    run = proj / "runs" / "a" / ".lobster_workspace"
    run.mkdir(parents=True)
    try:
        LaminCatalog(run, create=False)
        refused = None
    except CatalogUnavailable as exc:
        refused = str(exc)
    emit(root=root, refused=refused, key_root=str(own.layout.root))
"""


def test_an_old_project_catalog_serves_the_project_and_refuses_runs_with_the_fix(
    tmp_path, project
):
    run(OLD_CATALOG, tmp_path, project)
    out = run(OLD_USE, tmp_path, project)
    home = project / ".lobster_workspace"
    assert Path(out["root"]) == home == Path(out["key_root"])
    assert "mv " in out["refused"] and str(home / ".lamindb") in out["refused"]
    assert not (project / ".lamindb").exists()


def test_the_runs_of_a_project_with_an_old_catalog_are_not_recorded_and_say_why(
    tmp_path, project
):
    run(OLD_CATALOG, tmp_path, project)
    first = run(RUNS, tmp_path, project, RUNS="a")
    assert first["kinds"] == ["NullCatalog"]
    assert "mv " in first["reason"]
    assert not (project / ".lamindb").exists()
