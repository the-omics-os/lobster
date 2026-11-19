"""LaminDB engine behind the catalog port (runs only with the ``lamin`` extra).

LaminDB keeps process-wide settings, so every scenario runs in its own Python
process with an isolated HOME, a fresh current directory and every socket attempt
refused and logged. Each scenario prints one JSON line.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        importlib.util.find_spec("lamindb_setup") is None
        or importlib.util.find_spec("lamindb") is None,
        reason="needs the lamin extra (lamindb-core)",
    ),
]

PRELUDE = textwrap.dedent("""
    import json, os, shutil, sys
    NET = []
    def _hook(event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname"):
            addr = args[1] if event == "socket.connect" else (args[0] if args else None)
            if event == "socket.connect" and isinstance(addr, (str, bytes)):
                return
            NET.append(f"{event} {addr!r}")
            raise OSError("network blocked in test")
    sys.addaudithook(_hook)
    def emit(**kw):
        kw["net"] = NET
        print("RESULT " + json.dumps(kw, default=str), flush=True)
    """)


def _isolated_env(tmp_path, env_extra, drop_enable):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("LAMIN", "LOBSTER_LAMIN"))
    }
    env.update({"HOME": str(home), "LOBSTER_LAMIN": "1", "NO_PROXY": ""})
    env.update(
        {
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "XDG_CONFIG_HOME": str(home / ".config"),
        }
    )
    env.update(env_extra or {})
    if drop_enable:
        env.pop("LOBSTER_LAMIN", None)
    return env


def _run(
    code,
    tmp_path,
    *,
    env_extra=None,
    cwd=None,
    timeout=240,
    drop_enable=False,
    streams=False,
):
    env = _isolated_env(tmp_path, env_extra, drop_enable)
    cwd = Path(cwd or tmp_path / "cwd")
    cwd.mkdir(exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-c", PRELUDE + textwrap.dedent(code)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    marker = "RESULT "
    lines = [line for line in proc.stdout.splitlines() if line.startswith(marker)]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(lines[-1].removeprefix(marker))
    if streams:
        return result, proc.stdout, proc.stderr
    return result


def _tree(path):
    path = Path(path)
    if not path.exists():
        return []
    return sorted(str(p.relative_to(path)) for p in path.rglob("*"))


SESSION = """
    from pathlib import Path
    import anndata as ad, numpy as np
    from lobster.core.catalog import factory
    from lobster.core.runtime.data_manager import DataManagerV2

    ws = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ["WS"])
    def adata(seed):
        rng = np.random.default_rng(seed)
        a = ad.AnnData(rng.poisson(1.0, size=(40, 10)).astype(np.float32))
        a.obs_names = [f"c{i}" for i in range(40)]
        a.var_names = [f"g{i}" for i in range(10)]
        return a

    dm = DataManagerV2(workspace_path=ws)
    # An earlier session records the file a later session loads.
    catalog = factory.begin_session_run(dm, ws, "session_20261101_090000")
    dm.store_modality("input", adata(0))
    dm.save_modality("input", "input_copy.h5ad")
    factory.note_run_status(dm, "completed")
    sid = "session_20261108_100000"
    factory.begin_session_run(dm, ws, sid, ws / ".lobster" / "sessions" / sid)
    dm.load_modality(
        "loaded", dm.data_dir / "input_copy.h5ad",
        adapter="transcriptomics_single_cell", validate=False,
    )

    dm.store_modality("x", adata(1), step_summary="first version")
    v1 = Path(dm.save_modality("x", "x.h5ad"))
    v1_bytes = v1.read_bytes()
    dm.store_modality("x", adata(2), step_summary="second version")
    v2 = Path(dm.save_modality("x", "x.h5ad"))
    nb = ws / "notebooks" / "session.ipynb"
    nb.parent.mkdir(parents=True, exist_ok=True)
    nb.write_text(json.dumps({"cells": [], "nbformat": 4, "nbformat_minor": 5, "metadata": {}}))
    dm.log_tool_usage("qc", {}, description="d")
    from unittest.mock import Mock
    dm._notebook_exporter = Mock(export=Mock(return_value=nb))
    dm.export_notebook("session", description="exported session")
    status = catalog.status()
    catalog.close("completed")
    kept = ws / ".lamindb" / "versions" / "data" / "x.h5ad" / "v0000.h5ad"
    emit(type=type(catalog).__name__, degraded=status.degraded, reason=status.reason,
         mode=status.mode, instance=status.instance, location=status.location,
         v1=str(v1), v2=str(v2), v1_kept=(kept.is_file() and kept.read_bytes() == v1_bytes),
         data_files=sorted(p.name for p in (ws / "data").iterdir()))
"""

READBACK = """
    from pathlib import Path
    ws = Path(os.environ["WS"])
    os.environ["LAMIN_SETTINGS_DIR"] = str(ws / ".lamindb" / "settings")
    import lamindb_setup as lns
    slug = json.loads((ws / ".lamindb" / "lobster.json").read_text())["slug"]
    lns.connect(slug, _write_settings=False)
    import lamindb as ln
    run = ln.Run.get(name="session_20261108_100000")
    earlier = ln.Run.get(name="session_20261101_090000")
    fam = list(ln.Artifact.filter(key="data/x.h5ad").order_by("created_at"))
    tr = ln.Transform.get(key="notebooks/session.ipynb")
    emit(slug=slug, run_status=run.status, run_transform=run.transform.key,
         run_transform_version=run.transform.version,
         outputs=sorted({a.key for a in run.output_artifacts.all()}),
         inputs=sorted({a.key for a in run.input_artifacts.all()}),
         x_versions=[[a.version, a.is_latest, a._real_key, a.description] for a in fam],
         x_paths=[str(a.path) for a in fam],
         notebook_run_is_session=(tr.run_id == run.id), notebook_kind=tr.kind,
         earlier_status=earlier.status, run_reference_type=run.reference_type,
         run_extra=run.extra_data)
"""


@pytest.fixture(scope="module")
def session_result(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lamin_session")
    ws = tmp / "ws"
    ws.mkdir()
    (ws / "README.txt").write_text("pre-existing\n")
    home_before = _tree(tmp / "home")
    first = _run(SESSION, tmp, env_extra={"WS": str(ws)})
    readback = _run(READBACK, tmp, env_extra={"WS": str(ws)})
    return {
        "tmp": tmp,
        "ws": ws,
        "first": first,
        "readback": readback,
        "home_before": home_before,
    }


def test_session_recorded_without_network(session_result):
    first = session_result["first"]
    assert first["type"] == "ThreadedCatalog"
    assert not first["degraded"], first["reason"]
    assert first["mode"] == "workspace" and first["instance"].startswith(
        "anonymous/lobster-ws-"
    )
    assert Path(first["location"]) == session_result["ws"].resolve()
    assert first["net"] == [] and session_result["readback"]["net"] == []


def test_name_holds_the_latest_and_older_versions_are_kept(session_result):
    first, back = session_result["first"], session_result["readback"]
    ws = session_result["ws"].resolve()
    assert Path(first["v1"]).name == Path(first["v2"]).name == "x.h5ad"
    assert first["data_files"] == ["input_copy.h5ad", "x.h5ad"]
    assert first["v1_kept"]
    assert [v[:3] for v in back["x_versions"]] == [
        ["0000", False, ".lamindb/versions/data/x.h5ad/v0000.h5ad"],
        ["0001", True, None],
    ]
    assert [Path(p).resolve() for p in back["x_paths"]] == [
        ws / ".lamindb/versions/data/x.h5ad/v0000.h5ad",
        ws / "data/x.h5ad",
    ]
    assert [v[3] for v in back["x_versions"]] == ["first version", "second version"]


def test_run_inputs_outputs_and_notebook(session_result):
    back = session_result["readback"]
    assert back["run_status"] == "completed"
    assert back["run_transform"] == "lobster-ai"
    assert back["outputs"] == ["data/x.h5ad"]
    assert back["earlier_status"] == "completed"
    assert back["inputs"] == ["data/input_copy.h5ad"]
    assert back["notebook_run_is_session"] and back["notebook_kind"] == "notebook"
    assert back["run_reference_type"] == "lobster_session"
    assert back["run_extra"]["session_id"] == "session_20261108_100000"
    assert back["run_extra"]["kind"] == "session"


def test_nothing_written_outside_the_workspace(session_result):
    tmp = session_result["tmp"]
    assert _tree(tmp / "home") == session_result["home_before"]
    assert _tree(tmp / "cwd") == []
    ws_tree = _tree(session_result["ws"])
    assert (
        "README.txt" in ws_tree
        and ".lamindb/lamin.db" in ws_tree
        and ".lamindb/lobster.json" in ws_tree
    )
    assert any(
        p.startswith(".lamindb/settings/.lamin/instance--anonymous--") for p in ws_tree
    )


def test_identical_resave_returns_existing_and_keeps_description(tmp_path):
    code = """
        from pathlib import Path
        from lobster.core.catalog.lamin import LaminCatalog
        from lobster.core.catalog.port import SessionInfo
        ws = Path(os.environ["WS"]).resolve()
        (ws / "data").mkdir(parents=True)
        a = ws / "data" / "a.csv"
        a.write_text("x,y\\n1,2\\n")
        eng = LaminCatalog(ws)
        eng.begin_run(SessionInfo("s1", ws, "0.0.0"))
        kw = dict(key="data/a.csv", kind="dataset", otype=None, meta={})
        first = eng.register_artifact(a, description="original", **kw)
        target = eng.keep_path(a)  # names where the recorded version is kept
        untouched = a.is_file() and not target.exists()  # the engine moves nothing
        target.parent.mkdir(parents=True)
        shutil.copy2(a, target)  # what the saving thread does
        a.write_text("x,y\\n1,2\\n")  # the same bytes again
        again = eng.register_artifact(a, description="changed", **kw)
        art = eng.ln.Artifact.get(first.uid)
        unrecorded = eng.keep_path(ws / "data/b.csv")
        # Already overwritten: there is nothing left to keep.
        a.write_text("x,y\\n9,9\\n")
        after_overwrite = eng.keep_path(a)
        emit(target=str(target.relative_to(ws)), untouched=untouched,
             again_existing=again.existing,
             again_uid=again.uid, first_uid=first.uid, again_path=str(again.path),
             description=art.description, record_path=str(art.path),
             n=eng.ln.Artifact.filter(key="data/a.csv").count(), unrecorded=unrecorded,
             after_overwrite=after_overwrite, a_still_there=a.is_file(),
             versions=sorted(str(p.relative_to(ws)) for p in (ws / ".lamindb").rglob("*.csv")))
    """
    ws = tmp_path / "ws"
    r = _run(code, tmp_path, env_extra={"WS": str(ws)})
    assert r["target"] == ".lamindb/versions/data/a.csv/v0000.csv" and r["untouched"]
    assert r["again_existing"] and r["again_uid"] == r["first_uid"]
    assert Path(r["again_path"]).name == Path(r["record_path"]).name == "a.csv"
    assert r["description"] == "original" and r["n"] == 1
    assert r["versions"] == []  # the kept copy went back to its name
    assert r["unrecorded"] is None
    assert r["after_overwrite"] is None and r["a_still_there"]
    assert r["net"] == []


def test_same_bytes_under_another_name_move_a_record_off_a_changed_file(tmp_path):
    """A hash match whose recorded file no longer holds those bytes is not trusted.

    a.csv is recorded with bytes A, then rewritten with B while the catalog could
    not prepare the save (B is recorded at the same file). Saving A again as b.csv
    finds the record of A, whose own file now holds B: that record moves to b.csv
    and the save is not reported as existing.
    """
    code = """
        from pathlib import Path
        from lobster.core.catalog.lamin import LaminCatalog
        from lobster.core.catalog.port import SessionInfo
        ws = Path(os.environ["WS"]).resolve()
        (ws / "data").mkdir(parents=True)
        a, b = ws / "data" / "a.csv", ws / "data" / "b.csv"
        bytes_a, bytes_b = "a,b\\n1,2\\n", "a,b\\n3,4\\n"
        eng = LaminCatalog(ws)
        eng.begin_run(SessionInfo("s1", ws, "0.0.0"))
        a.write_text(bytes_a)
        ra = eng.register_artifact(a, key="data/a.csv", kind="dataset", otype=None,
                                   description="A", meta={})
        a.write_text(bytes_b)  # no keep_path: the start-up window
        rb = eng.register_artifact(a, key="data/a.csv", kind="dataset", otype=None,
                                   description="B", meta={})
        stale = eng.ln.Artifact.get(ra.uid)
        b.write_text(bytes_a)
        again = eng.register_artifact(b, key="data/b.csv", kind="dataset", otype=None,
                                      description="A again", meta={})
        moved = eng.ln.Artifact.get(ra.uid)
        emit(rb_existing=rb.existing, stale_tag=(stale.extra_data or {}).get("lobster"),
             again_uid=again.uid, ra_uid=ra.uid, again_existing=again.existing,
             again_path=Path(str(again.path)).name, moved_path=Path(str(moved.path)).name,
             moved_tag=moved.extra_data["lobster"], a_holds_b=(a.read_text() == bytes_b),
             b_kept=b.is_file())
    """
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert not r["rb_existing"]
    assert r["stale_tag"]["superseded_by"]
    assert r["again_uid"] == r["ra_uid"] and not r["again_existing"]
    assert r["again_path"] == r["moved_path"] == "b.csv" and r["b_kept"]
    assert r["moved_tag"]["relocated_from"] == "data/a.csv"
    assert "superseded_by" not in r["moved_tag"] and r["a_holds_b"]
    assert r["net"] == []


ADATA = """
    import anndata as ad, numpy as np
    def adata(seed):
        rng = np.random.default_rng(seed)
        a = ad.AnnData(rng.poisson(1.0, size=(30, 6)).astype(np.float32))
        a.obs_names = [f"c{i}" for i in range(30)]
        a.var_names = [f"g{i}" for i in range(6)]
        return a
"""

THREE_SAVES = ADATA + """
    import hashlib
    from pathlib import Path
    from lobster.core.catalog import factory
    from lobster.core.runtime.data_manager import DataManagerV2
    ws = Path(os.environ["WS"]).resolve()
    dm = DataManagerV2(workspace_path=ws)
    catalog = factory.begin_session_run(dm, ws, "session_a")
    returned = []
    for seed in (1, 2, 3):
        dm.modalities["x"] = adata(seed)
        returned.append(Path(dm.save_modality("x", "x.h5ad")).name)
    st = catalog.status()
    getattr(catalog, "close", lambda status: None)("completed")
    data = ws / "data"
    emit(catalog=type(catalog).__name__, degraded=st.degraded, reason=st.reason,
         returned=returned,
         data_files=sorted(str(p.relative_to(data)) for p in data.rglob("*")),
         canonical=hashlib.sha256((data / "x.h5ad").read_bytes()).hexdigest())
"""

NEXT_PROCESS = ADATA + """
    import anndata as ad, numpy as np
    from pathlib import Path
    from lobster.core.catalog import factory
    from lobster.core.runtime.data_manager import DataManagerV2
    ws = Path(os.environ["WS"]).resolve()
    dm = DataManagerV2(workspace_path=ws)
    catalog = factory.begin_command_run(dm, ws, "session_a")
    loaded = ad.read_h5ad(ws / "data" / "x.h5ad")
    store = ws / ".lamindb" / "versions"
    kept = sorted(p for p in store.rglob("*.h5ad") if p.is_file()) if store.exists() else []
    kept_match = [
        bool(np.array_equal(ad.read_h5ad(p).X, adata(seed).X)) for p, seed in zip(kept, (1, 2))
    ]
    scan = sorted(dm._scan_workspace())
    getattr(catalog, "close", lambda status: None)("completed")
    emit(latest=bool(np.array_equal(loaded.X, adata(3).X)),
         kept=[str(p.relative_to(ws)) for p in kept], kept_match=kept_match, scan=scan)
"""


PRUNE = """
    import hashlib
    from pathlib import Path
    from lobster.core.catalog.lamin import LaminCatalog
    ws = Path(os.environ["WS"]).resolve()
    eng = LaminCatalog(ws)
    store = ws / ".lamindb" / "versions"
    canonical = ws / "data" / "x.h5ad"

    def kept_files():
        return sorted(str(p.relative_to(ws)) for p in store.rglob("*") if p.is_file())

    def digest():
        return hashlib.sha256(canonical.read_bytes()).hexdigest()

    before, canonical_before = kept_files(), digest()
    sizes = [(ws / name).stat().st_size for name in before]
    plan = eng.kept_versions()
    result = eng.prune_versions(plan)
    rows = {r["version"]: r for r in eng.list_artifacts()}
    emit(before=before, sizes=sizes, plan=plan, result=result,
         after=kept_files(), store_stays=store.is_dir(), folders=sorted(
             str(p.relative_to(ws)) for p in store.rglob("*") if p.is_dir()),
         canonical_same=(digest() == canonical_before),
         notes={v: r["notes"] for v, r in rows.items()},
         latest=[v for v, r in rows.items() if r["latest"]],
         again=eng.kept_versions(), again_result=eng.prune_versions([]))
"""

BACK_AFTER_PRUNE = ADATA + """
    from pathlib import Path
    from lobster.core.catalog import factory
    from lobster.core.runtime.data_manager import DataManagerV2
    ws = Path(os.environ["WS"]).resolve()
    dm = DataManagerV2(workspace_path=ws)
    catalog = factory.begin_session_run(dm, ws, "session_b")
    dm.modalities["x"] = adata(1)  # the bytes of the first version, whose copy was pruned
    dm.save_modality("x", "x.h5ad")
    st = catalog.status()
    engine = catalog._engine
    rows = {r["version"]: r for r in engine.list_artifacts()}
    catalog.close("completed")
    emit(degraded=st.degraded, reason=st.reason,
         latest=[v for v, r in rows.items() if r["latest"]],
         notes={v: r["notes"] for v, r in rows.items()},
         files={v: r["file"] for v, r in rows.items()},
         data_files=sorted(p.name for p in (ws / "data").iterdir()),
         kept=sorted(str(p.relative_to(ws))
                     for p in (ws / ".lamindb" / "versions").rglob("*") if p.is_file()))
"""


def test_prune_removes_the_kept_versions_and_keeps_their_records(tmp_path):
    env = {"WS": str(tmp_path / "ws")}
    first = _run(THREE_SAVES, tmp_path, env_extra=env)
    assert not first["degraded"], first["reason"]
    r = _run(PRUNE, tmp_path, env_extra=env)
    kept = ".lamindb/versions/data/x.h5ad/"
    assert r["before"] == [kept + "v0000.h5ad", kept + "v0001.h5ad"]
    assert [(row["key"], row["version"], row["file"]) for row in r["plan"]] == [
        ("data/x.h5ad", "0000", kept + "v0000.h5ad"),
        ("data/x.h5ad", "0001", kept + "v0001.h5ad"),
    ]
    assert [row["bytes"] for row in r["plan"]] == r["sizes"]
    assert r["result"]["failed"] == []
    assert [row["version"] for row in r["result"]["removed"]] == ["0000", "0001"]
    assert r["after"] == [] and r["folders"] == []  # no empty folders left behind
    assert r["store_stays"]  # but the store itself is kept
    assert r["canonical_same"] and r["latest"] == ["0002"]
    assert r["notes"] == {"0000": ["not_kept"], "0001": ["not_kept"], "0002": []}
    assert r["again"] == []
    assert r["again_result"] == {"removed": [], "failed": [], "skipped": []}
    assert r["net"] == []

    back = _run(BACK_AFTER_PRUNE, tmp_path, env_extra=env)
    assert not back["degraded"], back["reason"]
    # The first version's bytes are saved again: it is the file name's record, with
    # no note left, and the replaced version is the one kept aside now.
    assert back["latest"] == ["0000"]
    assert back["notes"] == {"0000": [], "0001": ["not_kept"], "0002": []}
    assert back["files"]["0000"] == "data/x.h5ad"
    assert back["data_files"] == ["x.h5ad"]
    assert back["kept"] == [kept + "v0002.h5ad"]
    assert back["net"] == []


PRUNE_REFUSED = """
    import stat
    from pathlib import Path
    from lobster.core.catalog.lamin import LaminCatalog
    ws = Path(os.environ["WS"]).resolve()
    eng = LaminCatalog(ws)
    folder = ws / ".lamindb" / "versions" / "data" / "x.h5ad"
    plan = eng.kept_versions()
    (folder / "v0001.h5ad").rename(ws / "moved.h5ad")  # gone since it was shown
    folder.chmod(0o500)  # v0000 cannot be removed
    try:
        result = eng.prune_versions(plan)
    finally:
        folder.chmod(0o700)
    rows = {r["version"]: r for r in eng.list_artifacts()}
    emit(result=result, notes={v: r["notes"] for v, r in rows.items()},
         left=sorted(p.name for p in folder.iterdir()))
"""


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root removes whatever the folder mode",
)
def test_prune_marks_only_the_versions_it_removed(tmp_path):
    env = {"WS": str(tmp_path / "ws")}
    _run(THREE_SAVES, tmp_path, env_extra=env)
    r = _run(PRUNE_REFUSED, tmp_path, env_extra=env)
    assert r["result"]["removed"] == []  # v0001's file is gone, so it is not listed
    assert r["result"]["failed"] == [
        {"file": ".lamindb/versions/data/x.h5ad/v0000.h5ad", "error": "PermissionError"}
    ]
    assert [x["file"] for x in r["result"]["skipped"]] == [
        ".lamindb/versions/data/x.h5ad/v0001.h5ad"
    ]
    # v0000 was marked before its removal failed: the mark is taken back
    assert r["notes"] == {"0000": [], "0001": [], "0002": []}
    assert r["left"] == ["v0000.h5ad"]
    assert r["net"] == []


PRUNE_FOLD = """
    import sys
    from pathlib import Path
    from lobster.core.catalog.lamin import (
        CatalogBusy, CatalogReadOnly, LaminCatalog)
    from lobster.cli_internal.commands.light.lamin_commands import overview
    ws = Path(os.environ["WS"]).resolve()
    case = os.environ["CASE"]
    db = ws / ".lamindb" / "lamin.db"
    if case == "readonly":  # opened read-only, as a file the user may not write is
        db.chmod(0o444)
    eng = LaminCatalog(ws)
    store = ws / ".lamindb" / "versions" / "data" / "x.h5ad"

    def files():
        return sorted(p.name for p in store.iterdir()) if store.is_dir() else []

    def notes():
        return {r["version"]: r["notes"] for r in eng.list_artifacts()}

    plan = eng.kept_versions()
    out = dict(plan=[r["version"] for r in plan])
    if case == "busy":
        try:
            eng.prune_versions(plan)
        except CatalogBusy as exc:
            out["raised"] = type(exc).__name__
    elif case == "readonly":
        try:
            eng.prune_versions(plan)
        except CatalogReadOnly as exc:
            out["raised"] = type(exc).__name__
        finally:
            db.chmod(0o644)
    elif case == "unmarked":
        def refuse(record, name, value):
            if record.version == "0000":
                raise RuntimeError("cannot save")
            return type(eng)._tag_strict(eng, record, name, value)
        eng._tag_strict = refuse
        out["result"] = eng.prune_versions(plan)
    elif case == "save_fails":  # the probe's save works, the marks do not
        saves = []
        real = type(eng.ln.Artifact.filter().first()).save
        def save(self, *a, **k):
            saves.append(1)
            if len(saves) > 1:
                raise RuntimeError("disk full")
            return real(self, *a, **k)
        type(eng.ln.Artifact.filter().first()).save = save
        try:
            out["result"] = eng.prune_versions(plan)
        finally:
            type(eng.ln.Artifact.filter().first()).save = real
    elif case == "dup":
        out["result"] = eng.prune_versions([plan[0], plan[0], plan[1], plan[0]])
    elif case == "vanishes":  # the file goes between the offer and the removal
        real = Path.unlink
        def unlink(self, *a, **k):
            real(self, *a, **k)
            if self.name == "v0000.h5ad":
                raise FileNotFoundError(self)
        Path.unlink = unlink
        try:
            out["result"] = eng.prune_versions(plan)
        finally:
            Path.unlink = real
    elif case == "shown":
        out["result"] = eng.prune_versions(
            [plan[0], dict(plan[1], uid="not-a-uid")])
    elif case == "symlink":
        outside = Path(os.environ["OUTSIDE"])
        (store / "v0000.h5ad").unlink()
        (store / "v0000.h5ad").symlink_to(outside)
        out["listed"] = [r["version"] for r in eng.kept_versions()]
        out["result"] = eng.prune_versions(plan)
        out["outside_kept"] = outside.is_file()
        out["link_kept"] = (store / "v0000.h5ad").is_symlink()
    elif case == "current":
        latest = [r for r in eng.list_artifacts() if r["latest"]][0]
        from django.db.models import Q
        eng.ln.Artifact.filter(Q(uid=plan[1]["uid"])).update(
            _real_key=".lamindb/versions/data/x.h5ad/gone.h5ad")
        eng.ln.Artifact.filter(Q(uid=latest["uid"])).update(
            _real_key=".lamindb/versions/data/x.h5ad/v0001.h5ad")
        out["listed"] = [r["version"] for r in eng.kept_versions()]
        out["result"] = eng.prune_versions(plan)
    elif case == "same_set":
        (store / "stray.bin").write_bytes(b"s" * 999)  # nobody's file
        shown = eng.kept_versions()
        status = overview(ws)["versions"]
        removed = eng.prune_versions(shown)["removed"]
        out.update(status=status, shown=sum(r["bytes"] for r in shown),
                   removed=sum(r["bytes"] for r in removed),
                   files=len(removed), stray=(store / "stray.bin").is_file())
    out.update(files=files(), notes=notes())
    emit(**out)
"""


def _prune_fold(tmp_path, case, hold=False, **env_extra):
    env = {"WS": str(tmp_path / "ws"), "CASE": case, **env_extra}
    _run(THREE_SAVES, tmp_path, env_extra=env)
    if not hold:
        return _run(PRUNE_FOLD, tmp_path, env_extra=env)
    from lobster.core.catalog.thread import workspace_lock

    with workspace_lock(tmp_path / "ws" / ".lobster" / "catalog.lock"):
        return _run(PRUNE_FOLD, tmp_path, env_extra=env)


V = ["v0000.h5ad", "v0001.h5ad"]


def test_prune_refuses_while_another_process_holds_the_catalog_lock(tmp_path):
    r = _prune_fold(tmp_path, "busy", hold=True)
    assert r["raised"] == "CatalogBusy"
    assert r["files"] == V and set(map(tuple, r["notes"].values())) == {()}


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes anyway"
)
def test_prune_refuses_a_catalog_that_cannot_be_written_before_removing_anything(
    tmp_path,
):
    r = _prune_fold(tmp_path, "readonly")
    assert r["raised"] == "CatalogReadOnly"
    assert r["files"] == V and set(map(tuple, r["notes"].values())) == {()}


def test_a_file_whose_record_cannot_be_marked_is_left_and_listed(tmp_path):
    r = _prune_fold(tmp_path, "unmarked")
    assert [x["version"] for x in r["result"]["removed"]] == ["0001"]
    assert [x["file"] for x in r["result"]["failed"]] == [
        ".lamindb/versions/data/x.h5ad/v0000.h5ad"
    ]
    assert r["files"] == ["v0000.h5ad"]  # kept, and its record says nothing
    assert r["notes"] == {"0000": [], "0001": ["not_kept"], "0002": []}


def test_nothing_is_removed_when_every_mark_fails_to_save(tmp_path):
    r = _prune_fold(tmp_path, "save_fails")
    assert r["result"]["removed"] == []
    assert [x["error"] for x in r["result"]["failed"]] == ["NotMarked"] * 2
    assert r["files"] == V and set(map(tuple, r["notes"].values())) == {()}


def test_a_row_shown_twice_is_removed_once_and_stays_marked(tmp_path):
    r = _prune_fold(tmp_path, "dup")
    assert [x["version"] for x in r["result"]["removed"]] == ["0000", "0001"]
    assert r["result"]["failed"] == [] and r["result"]["skipped"] == []
    assert r["files"] == []
    assert r["notes"] == {"0000": ["not_kept"], "0001": ["not_kept"], "0002": []}


def test_a_file_that_is_already_gone_when_removed_counts_as_removed_and_stays_marked(
    tmp_path,
):
    r = _prune_fold(tmp_path, "vanishes")
    assert [x["version"] for x in r["result"]["removed"]] == ["0000", "0001"]
    assert r["result"]["failed"] == []
    assert r["notes"] == {"0000": ["not_kept"], "0001": ["not_kept"], "0002": []}


def test_prune_removes_exactly_what_was_shown(tmp_path):
    r = _prune_fold(tmp_path, "shown")
    assert [x["version"] for x in r["result"]["removed"]] == ["0000"]
    assert [x["file"] for x in r["result"]["skipped"]] == [
        ".lamindb/versions/data/x.h5ad/v0001.h5ad"
    ]
    assert r["files"] == ["v0001.h5ad"]  # not shown as it is now: kept
    assert r["notes"]["0001"] == []


def test_a_link_to_a_file_outside_the_store_is_never_removed(tmp_path):
    outside = tmp_path / "outside.h5ad"
    outside.write_bytes(b"mine")
    r = _prune_fold(tmp_path, "symlink", OUTSIDE=str(outside))
    assert r["listed"] == ["0001"]  # not offered
    assert [x["version"] for x in r["result"]["removed"]] == ["0001"]
    assert [x["file"] for x in r["result"]["skipped"]] == [
        ".lamindb/versions/data/x.h5ad/v0000.h5ad"
    ]
    assert r["outside_kept"] and r["link_kept"]
    assert r["notes"]["0000"] == []


def test_the_file_of_a_current_version_is_never_removed(tmp_path):
    # the latest record now owns v0001.h5ad; the record that did is offered no more
    r = _prune_fold(tmp_path, "current")
    assert r["listed"] == ["0000"]
    assert [x["version"] for x in r["result"]["removed"]] == ["0000"]
    assert [x["file"] for x in r["result"]["skipped"]] == [
        ".lamindb/versions/data/x.h5ad/v0001.h5ad"
    ]
    assert r["files"] == ["v0001.h5ad"] and r["notes"]["0002"] == []


def test_status_and_prune_count_the_same_files(tmp_path):
    r = _prune_fold(tmp_path, "same_set")
    assert r["status"] == {"files": 2, "bytes": r["shown"]}
    assert r["removed"] == r["shown"] and r["files"] == ["stray.bin"]
    assert r["stray"]  # the stray file is neither counted nor removed


def test_catalog_on_and_off_leave_the_same_files(tmp_path):
    """The file name holds the latest save whether or not the catalog is on."""
    results = {}
    for flag in ("1", "0"):
        tmp = tmp_path / f"lamin_{flag}"
        tmp.mkdir()
        env = {"WS": str(tmp / "ws"), "LOBSTER_LAMIN": flag}
        results[flag] = (
            _run(THREE_SAVES, tmp, env_extra=env),
            _run(NEXT_PROCESS, tmp, env_extra=env),
        )
    (on, on_next), (off, off_next) = results["1"], results["0"]
    assert on["catalog"] == "ThreadedCatalog" and not on["degraded"], on["reason"]
    assert off["catalog"] == "NullCatalog"
    assert on["returned"] == off["returned"] == ["x.h5ad"] * 3
    assert on["data_files"] == off["data_files"] == ["x.h5ad"]
    assert on["canonical"] == off["canonical"]
    assert on_next["latest"] and off_next["latest"]
    assert on_next["scan"] == off_next["scan"] == ["x"]
    assert on_next["kept"] == [
        ".lamindb/versions/data/x.h5ad/v0000.h5ad",
        ".lamindb/versions/data/x.h5ad/v0001.h5ad",
    ]
    assert on_next["kept_match"] == [True, True] and off_next["kept"] == []
    assert on["net"] == on_next["net"] == []

    readback = """
        from pathlib import Path
        ws = Path(os.environ["WS"])
        os.environ["LAMIN_SETTINGS_DIR"] = str(ws / ".lamindb" / "settings")
        import lamindb_setup as lns
        lns.connect(json.loads((ws / ".lamindb/lobster.json").read_text())["slug"],
                    _write_settings=False)
        import lamindb as ln
        runs = ln.Run.filter(name="session_a").order_by("started_at")
        emit(runs=[[r.reference_type, (r.extra_data or {}).get("kind")] for r in runs],
             versions=[[a.version, a.is_latest] for a in
                       ln.Artifact.filter(key="data/x.h5ad").order_by("created_at")])
    """
    back = _run(
        readback, tmp_path / "lamin_1", env_extra={"WS": str(tmp_path / "lamin_1/ws")}
    )
    assert back["runs"] == [
        ["lobster_session", "session"],
        ["lobster_command", "command"],
    ]
    assert back["versions"] == [["0000", False], ["0001", False], ["0002", True]]


@pytest.mark.parametrize("stall_in", ["_hash_or_none", "_version_file"])
def test_a_keep_answered_too_late_moves_nothing(tmp_path, stall_in):
    """The engine stalls before or after it decided, past the caller's wait.

    The caller writes the new bytes at the name without keeping anything or
    recording the save; the engine, when it answers, changes no file. The old
    record is marked.
    """
    code = textwrap.dedent(ADATA) + textwrap.dedent(f"""
        import time
        import anndata as ad, numpy as np
        from pathlib import Path
        from lobster.core.catalog import factory, lamin, thread
        from lobster.core.runtime.data_manager import DataManagerV2
        thread.KEEP_TIMEOUT_S = 0.5
        ws = Path(os.environ["WS"]).resolve()
        dm = DataManagerV2(workspace_path=ws)
        catalog = factory.begin_session_run(dm, ws, "session_a")
        dm.modalities["x"] = adata(1)
        x = Path(dm.save_modality("x", "x.h5ad"))
        original = getattr(lamin.LaminCatalog, "{stall_in}")
        def stalled(self, *args, **kwargs):
            time.sleep(2)
            return original(self, *args, **kwargs)
        setattr(lamin.LaminCatalog, "{stall_in}", stalled)
        dm.modalities["x"] = adata(2)
        t0 = time.monotonic()
        out = Path(dm.save_modality("x", "x.h5ad"))
        waited = time.monotonic() - t0
        right_after = bool(np.array_equal(ad.read_h5ad(x).X, adata(2).X))
        setattr(lamin.LaminCatalog, "{stall_in}", original)
        time.sleep(3)  # the stalled call has answered
        later = bool(np.array_equal(ad.read_h5ad(x).X, adata(2).X))
        store = ws / ".lamindb" / "versions"
        kept = sorted(str(p) for p in store.rglob("*") if p.is_file()) if store.exists() else []
        records = {{
            a.version: (a.extra_data or {{}}).get("lobster", {{}})
            for a in catalog._engine.ln.Artifact.filter(key="data/x.h5ad")
        }}
        dm.modalities["x"] = adata(3)  # the next save is recorded again
        dm.save_modality("x", "x.h5ad")
        st = catalog.status()
        latest = [
            (a.version, a.hash == lamin._lamin_hash(x))
            for a in catalog._engine.ln.Artifact.filter(key="data/x.h5ad", is_latest=True)
        ]
        catalog.close("completed")
        emit(out=str(out.relative_to(ws)), waited=waited, right_after=right_after,
             later=later, kept=kept, degraded=st.degraded, reason=st.reason,
             records=records, latest=latest)
    """)
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert r["out"] == "data/x.h5ad" and r["waited"] < 30
    assert r["right_after"] and r["later"], "the name lost the latest bytes"
    assert r["kept"] == []
    assert not r["degraded"], r["reason"]
    # Not recorded (the catalog was busy), and the old record no longer claims
    # the file.
    assert list(r["records"]) == ["0000"]
    assert r["records"]["0000"]["superseded_by"] == "unrecorded"
    assert r["latest"] == [["0001", True]]  # one latest per key: the file's bytes
    assert r["net"] == []


def test_saving_an_older_version_again_makes_it_the_latest(tmp_path):
    """A, B, then A again: the record of A is back at the name and is the latest."""
    code = textwrap.dedent(ADATA) + textwrap.dedent("""
        from pathlib import Path
        from lobster.core.catalog import factory, lamin
        from lobster.core.runtime.data_manager import DataManagerV2
        ws = Path(os.environ["WS"]).resolve()
        dm = DataManagerV2(workspace_path=ws)
        catalog = factory.begin_session_run(dm, ws, "session_a")
        for seed in (1, 2, 1):
            dm.modalities["x"] = adata(seed)
            x = Path(dm.save_modality("x", "x.h5ad"))
        st = catalog.status()
        eng = catalog._engine
        rows = [
            (a.version, a.is_latest, eng._storage_key(eng._stored_path(a)),
             a.hash == lamin._lamin_hash(eng._stored_path(a)))
            for a in eng.ln.Artifact.filter(key="data/x.h5ad").order_by("created_at")
        ]
        catalog.close("completed")
        emit(degraded=st.degraded, reason=st.reason, rows=rows)
    """)
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert not r["degraded"], r["reason"]
    assert r["rows"] == [
        ["0000", True, "data/x.h5ad", True],
        ["0001", False, ".lamindb/versions/data/x.h5ad/v0001.h5ad", True],
    ]
    assert r["net"] == []


# LaminDB hashes only the first and the last chunk of a file larger than its chunk
# (50 MiB), so two such files that differ in between have the same hash. The chunk
# is made 16 bytes here instead of writing 100 MB.
BIG_FILES = """
    import hashlib
    from pathlib import Path
    from lobster.core.catalog import factory
    from lobster.core.runtime.data_manager import DataManagerV2
    ws = Path(os.environ["WS"]).resolve()
    dm = DataManagerV2(workspace_path=ws)
    catalog = factory.begin_session_run(dm, ws, "session_a")
    catalog.status()  # the engine is up
    eng = catalog._engine
    import lamindb_setup.core.hashing as lamin_hashing
    lamin_hashing.hash_file.__defaults__ = (None, 16)
    path = ws / "data" / "big.csv"
    path.parent.mkdir(exist_ok=True)
    HEAD, TAIL = b"H" * 16, b"T" * 16

    def big(middle):
        return HEAD + middle + TAIL

    def save(middle):
        with catalog.save_section(path) as section:
            keep = dm._catalog_usable_keep(section.keep)
            with dm._catalog_keeping(path, keep, replaces=False):
                path.write_bytes(big(middle))
            return section.register(
                key="data/big.csv", kind="dataset", otype=None, description="d", meta={}
            )

    def state():
        rows, consistent = [], []
        for a in eng.ln.Artifact.filter(key="data/big.csv").order_by("created_at"):
            stored = eng._stored_path(a)
            rows.append([a.version, a.is_latest, eng._storage_key(stored)])
            recorded = ((a.extra_data or {}).get("lobster") or {}).get("sha256")
            consistent.append(
                stored.is_file()
                and recorded == hashlib.sha256(stored.read_bytes()).hexdigest()
            )
        return rows, consistent
"""


def test_files_that_differ_only_between_the_first_and_last_chunk_are_two_versions(
    tmp_path,
):
    """Same LaminDB hash, other bytes: a new version, kept and restored like any."""
    code = textwrap.dedent(BIG_FILES) + textwrap.dedent("""
        A, B = b"a" * 40, b"b" * 40
        steps = {}
        for label, middle in (("a", A), ("b", B), ("b_again", B), ("a_again", A)):
            ref = save(middle)
            rows, consistent = state()
            steps[label] = [ref.existing, rows, consistent]
        kept = sorted(
            p for p in (ws / ".lamindb" / "versions").rglob("*") if p.is_file()
        )
        st = catalog.status()
        catalog.close("completed")
        emit(degraded=st.degraded, reason=st.reason, steps=steps,
             file_is_a=(path.read_bytes() == big(A)),
             kept=[str(p.relative_to(ws)) for p in kept],
             data_files=sorted(p.name for p in (ws / "data").iterdir()))
    """)
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert not r["degraded"], r["reason"]
    kept = ".lamindb/versions/data/big.csv/"
    steps = r["steps"]
    assert [steps[k][0] for k in ("a", "b", "b_again", "a_again")] == [
        False,
        False,
        True,
        True,
    ]
    assert steps["a"][1] == [["0000", True, "data/big.csv"]]
    assert steps["b"][1] == [
        ["0000", False, kept + "v0000.csv"],
        ["0001", True, "data/big.csv"],
    ]
    assert steps["b_again"][1] == steps["b"][1]
    assert steps["a_again"][1] == [
        ["0000", True, "data/big.csv"],
        ["0001", False, kept + "v0001.csv"],
    ]
    # every record's file holds the bytes it records (the SHA-256 kept on the record)
    assert all(all(steps[k][2]) for k in steps)
    assert r["file_is_a"] and r["data_files"] == ["big.csv"]
    assert r["kept"] == [kept + "v0001.csv"]
    assert r["net"] == []


def test_a_record_without_a_sha256_is_judged_by_laminDBs_hash(tmp_path):
    """Records made before the SHA-256 was kept still match the same bytes."""
    code = textwrap.dedent(BIG_FILES) + textwrap.dedent("""
        first = save(b"a" * 40)
        record = eng.ln.Artifact.get(first.uid)
        record.extra_data = {"lobster": {}}
        record.save()
        again = save(b"a" * 40)
        rows, _ = state()
        catalog.close("completed")
        emit(existing=again.existing, rows=rows)
    """)
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert r["existing"] is True
    assert r["rows"] == [["0000", True, "data/big.csv"]]
    assert r["net"] == []


def test_a_file_with_the_same_first_and_last_chunk_is_not_linked_as_input(tmp_path):
    code = textwrap.dedent(BIG_FILES) + textwrap.dedent("""
        save(b"a" * 40)
        other = ws / "data" / "other.csv"
        other.write_bytes(big(b"b" * 40))
        same = ws / "data" / "same.csv"
        same.write_bytes(big(b"a" * 40))
        linked_other = eng.link_input(other)
        linked_same = eng.link_input(same)
        save(b"b" * 40)  # the second version has a hash of its own
        later = eng.link_input(other)
        catalog.close("completed")
        emit(other=linked_other and linked_other.key,
             same=linked_same and linked_same.key,
             later=later and later.version)
    """)
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert r["other"] is None and r["same"] == "data/big.csv"
    assert r["later"] == "0001"
    assert r["net"] == []


LOOK_ALIKES = """
    from pathlib import Path
    from lobster.core.catalog.lamin import LaminCatalog
    from lobster.core.catalog.port import SessionInfo
    ws = Path(os.environ["WS"]).resolve()
    (ws / "data").mkdir(parents=True)
    eng = LaminCatalog(ws)
    eng.begin_run(SessionInfo("s1", ws, "0.0.0"))
    import lamindb_setup.core.hashing as lamin_hashing  # after LaminDB is set up
    lamin_hashing.hash_file.__defaults__ = (None, 16)

    def big(middle):
        return b"H" * 16 + middle + b"T" * 16

    def kw(name):
        return dict(key=f"data/{name}", kind="dataset", otype=None,
                    description="d", meta={})
"""


def test_a_look_alike_written_while_the_catalog_was_busy_marks_the_record(tmp_path):
    code = textwrap.dedent(LOOK_ALIKES) + textwrap.dedent("""
        x = ws / "data" / "x.csv"
        x.write_bytes(big(b"a" * 40))
        r1 = eng.register_artifact(x, **kw("x.csv"))
        x.write_bytes(big(b"b" * 40))  # saved while the catalog was busy
        eng.note_unrecorded(x)
        tag = eng.ln.Artifact.get(r1.uid).extra_data["lobster"]
        emit(tag=tag)
    """)
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert r["tag"]["superseded_by"] == "unrecorded"
    assert r["net"] == []


def test_a_kept_copy_that_is_a_look_alike_is_not_adopted(tmp_path):
    code = textwrap.dedent(LOOK_ALIKES) + textwrap.dedent("""
        x = ws / "data" / "x.csv"
        x.write_bytes(big(b"a" * 40))
        r1 = eng.register_artifact(x, **kw("x.csv"))
        target = eng.keep_path(x)  # as the saving thread does before it writes
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(big(b"z" * 40))  # a copy with the same size and ends
        x.write_bytes(big(b"b" * 40))
        eng.register_artifact(x, **kw("x.csv"))
        record = eng.ln.Artifact.get(r1.uid)
        emit(tag=record.extra_data["lobster"],
             stored=eng._storage_key(eng._stored_path(record)))
    """)
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert r["tag"]["not_kept"] == "overwritten before it could be kept"
    assert r["stored"] == "data/x.csv"
    assert r["net"] == []


def test_a_recorded_file_that_became_a_look_alike_is_not_the_record_of_a_copy(tmp_path):
    code = textwrap.dedent(LOOK_ALIKES) + textwrap.dedent("""
        a, b = ws / "data" / "a.csv", ws / "data" / "b.csv"
        a.write_bytes(big(b"a" * 40))
        ra = eng.register_artifact(a, **kw("a.csv"))
        a.write_bytes(big(b"z" * 40))  # changed outside the catalog
        b.write_bytes(big(b"a" * 40))
        rb = eng.register_artifact(b, **kw("b.csv"))
        record = eng.ln.Artifact.get(ra.uid)
        emit(same_record=rb.uid == ra.uid,
             stored=eng._storage_key(eng._stored_path(record)),
             holds=eng._stored_path(record).read_bytes() == big(b"a" * 40))
    """)
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert r["same_record"] and r["stored"] == "data/b.csv" and r["holds"]
    assert r["net"] == []


def test_a_named_instance_keeps_a_copy_of_each_of_two_look_alike_files(tmp_path):
    """A named instance names the copy after the content: two files LaminDB hashes alike are two."""
    inst = tmp_path / "inst"
    _run(
        f"""
        import lamindb_setup
        lamindb_setup.init(storage={str(inst)!r}, name="mbbig")
        emit()
        """,
        tmp_path,
    )
    code = """
        from pathlib import Path
        from lobster.core.catalog.lamin import LaminCatalog
        from lobster.core.catalog.port import SessionInfo
        ws = Path(os.environ["WS"]).resolve()
        (ws / "data").mkdir(parents=True)
        x = ws / "data" / "x.csv"
        eng = LaminCatalog(ws, instance="anonymous/mbbig")
        eng.begin_run(SessionInfo("s1", ws, "0.0.0"))
        import lamindb_setup.core.hashing as lamin_hashing
        lamin_hashing.hash_file.__defaults__ = (None, 16)
        kw = dict(key="data/x.csv", kind="dataset", otype=None, description="d", meta={})
        head, tail = b"H" * 16, b"T" * 16
        refs = []
        for middle in (b"a" * 40, b"b" * 40, b"b" * 40):
            x.write_bytes(head + middle + tail)
            r = eng.register_artifact(x, **kw)
            refs.append([r.version, r.existing, r.path.read_bytes()[16:17].decode()])
        eng.end_run("completed")
        emit(refs=refs)
    """
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws"), "INST": str(inst)})
    assert r["refs"] == [
        ["0000", False, "a"],
        ["0001", False, "b"],
        ["0001", True, "b"],
    ]
    assert r["net"] == []


REPLAY = """
    from pathlib import Path
    from lobster.core.catalog.lamin import LaminCatalog
    from lobster.core.catalog.port import SessionInfo
    ws = Path(os.environ["WS"]).resolve()
    (ws / "data").mkdir(parents=True)
    eng = LaminCatalog(ws, os.environ.get("INST_SLUG"))
    NB = '{"cells": [], "metadata": {}}'
    NOTEBOOK = dict(key="notebooks/n.ipynb", kind="notebook", source=NB,
                    description="the export")
    PARAMS = {"input_data": "in.h5ad", "output_prefix": "out"}
    kw = lambda key: dict(key=key, kind="dataset", otype=None, description="d", meta={})
    shared = ws / "data" / "shared.h5ad"
    shared.write_bytes(b"shared bytes")
    eng.begin_run(SessionInfo("s0", ws, "0.0.0"))
    eng.register_artifact(shared, **kw("data/shared.h5ad"))
    eng.end_run("completed")
    eng.begin_run(SessionInfo("s1", ws, "0.0.0"))
    saved = ws / "data" / "pbmc.h5ad"
    saved.write_bytes(b"session bytes")
    eng.register_artifact(saved, **kw("data/pbmc.h5ad"))
    if os.environ.get("TWO_SAVES"):  # an earlier version of it is kept aside
        target = eng.keep_path(saved)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(saved, target)
        saved.write_bytes(b"final bytes")
        eng.register_artifact(saved, **kw("data/pbmc.h5ad"))
    if os.environ.get("XYX"):  # X, then Y, then X again: the file holds X
        for data in (b"bytes Y", b"session bytes"):
            target = eng.keep_path(saved)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(saved, target)
            saved.write_bytes(data)
            eng.register_artifact(saved, **kw("data/pbmc.h5ad"))
    if os.environ.get("ROOT_PBMC"):  # saved again, later, under the root name
        later = ws / "pbmc.h5ad"
        later.write_bytes(b"root pbmc")
        eng.register_artifact(later, **kw("pbmc.h5ad"))
    eng.register_artifact(shared, **kw("data/shared.h5ad"))  # the same bytes again
    root = ws / "root.h5ad"
    root.write_bytes(b"root bytes")
    eng.register_artifact(root, **kw("root.h5ad"))
    eng.register_transform(**NOTEBOOK)
    eng.end_run("completed")
    if os.environ.get("TWIN"):  # another session exported the same bytes, elsewhere
        eng.begin_run(SessionInfo("s3", ws, "0.0.0"))
        eng.register_transform(**{**NOTEBOOK, "key": "notebooks/m.ipynb"})
        eng.end_run("completed")
    session = eng.ln.Run.filter(name="s1").get()

    def replay(**given):
        fields = dict(key="notebooks/n.ipynb", source=NB, params=PARAMS,
                      lobster_version="0.0.0")
        return eng.begin_replay(**{**fields, **given})
"""


def test_a_replay_is_a_run_of_the_notebook_started_by_the_session(tmp_path):
    code = textwrap.dedent(REPLAY) + textwrap.dedent("""
        before = eng.ln.Transform.filter(kind="notebook").count()
        started = replay()
        run = eng.ln.Run.get(uid=started.uid)
        notebook = eng.ln.Transform.get(key="notebooks/n.ipynb")
        emit(session_uid=started.session_uid == session.uid,
             transform=run.transform.key,
             initiated=run.initiated_by_run_id == session.id,
             params=run.params, name=run.name, extra=run.extra_data,
             open=run.finished_at is None,
             new_transforms=eng.ln.Transform.filter(kind="notebook").count() - before,
             described=notebook.description, owner=notebook.run_id == session.id,
             sessions=eng.overview()["runs"], last=eng.overview()["last_run"]["uid"],
             listed=[r["uid"] for r in eng.list_runs() if r["kind"] == "session"],
             listed_all=len(eng.list_runs()), session_uid_=session.uid,
             untracked=eng.ln.context.run is None)
    """)
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert r["session_uid"] and r["transform"] == "notebooks/n.ipynb"
    assert r["initiated"] and r["open"]
    assert r["params"] == {"input_data": "in.h5ad", "output_prefix": "out"}
    assert r["name"] == "replay:n"
    assert r["extra"] == {
        "lobster_version": "0.0.0",
        "kind": "replay",
        "notebook": "notebooks/n.ipynb",
    }
    # the notebook is found by its source: no new Transform, its description and
    # its own run (the session) stay as the export left them
    assert r["new_transforms"] == 0 and r["described"] == "the export" and r["owner"]
    # a replay is not a session; `list_runs` lists it too, as a replay
    assert r["sessions"] == 2 and r["last"] == r["listed"][0] == r["session_uid_"]
    assert r["listed_all"] == len(r["listed"]) + 1
    assert r["untracked"] and r["net"] == []


def test_the_replay_reference_is_what_the_session_saved(tmp_path):
    code = textwrap.dedent(REPLAY) + textwrap.dedent("""
        # a later session saves other bytes at the same key and another name
        eng.begin_run(SessionInfo("s2", ws, "0.0.0"))
        saved.write_bytes(b"later bytes")
        eng.register_artifact(saved, **kw("data/pbmc.h5ad"))
        other = ws / "data" / "other.h5ad"
        other.write_bytes(b"other")
        eng.register_artifact(other, **kw("data/other.h5ad"))
        eng.end_run("completed")
        saved.write_bytes(b"session bytes")  # what the first session saved, again
        started = replay()
        found = eng.replay_reference("pbmc")
        emit(found=None if found is None else str(found.relative_to(ws)),
             content=None if found is None else found.read_bytes().decode(),
             other=eng.replay_reference("other"), missing=eng.replay_reference("nope"),
             unknown=eng.replay_reference("../x"),
             shared=str(eng.replay_reference("shared").relative_to(ws)),
             root=str(eng.replay_reference("root").relative_to(ws)))
    """)
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    # the first session's record, not the later session's newer one at the same key
    assert r["content"] == "session bytes"
    assert r["other"] is None and r["missing"] is None and r["unknown"] is None
    # bytes an earlier session saved and this one saved again; a file at the root
    assert r["shared"] == "data/shared.h5ad" and r["root"] == "root.h5ad"
    assert r["net"] == []


def test_the_replay_reference_is_none_when_the_recorded_file_changed(tmp_path):
    code = textwrap.dedent(REPLAY) + textwrap.dedent("""
        saved.write_bytes(b"changed outside Lobster")
        replay()
        emit(found=eng.replay_reference("pbmc"))
    """)
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert r["found"] is None and r["net"] == []


def test_the_reference_is_the_sessions_last_save_not_an_earlier_one(tmp_path):
    code = textwrap.dedent(REPLAY) + textwrap.dedent("""
        replay()
        intact = eng.replay_reference_state("pbmc")
        intact = [intact[0].read_bytes().decode(), intact[1]]
        saved.write_bytes(b"edited after the session")
        emit(intact=intact,
             changed=list(eng.replay_reference_state("pbmc")),
             plain=eng.replay_reference("pbmc"),
             none=list(eng.replay_reference_state("nope")),
             kept=sorted(p.name for p in (ws / ".lamindb" / "versions").rglob("*") if p.is_file()))
    """)
    env = {"WS": str(tmp_path / "ws"), "TWO_SAVES": "1"}
    r = _run(code, tmp_path, env_extra=env)
    assert r["kept"] == ["v0000.h5ad"]  # the earlier save is intact on disk
    assert r["intact"] == ["final bytes", "ok"]
    # the last save changed: the earlier one is not a stand-in for it
    assert r["changed"] == [None, "changed"] and r["plain"] is None
    assert r["none"] == [None, "none"]


def test_the_reference_is_the_record_with_the_bytes_of_the_workspace_file_not_the_newest(
    tmp_path,
):
    code = textwrap.dedent(REPLAY) + textwrap.dedent("""
        replay()
        found, state = eng.replay_reference_state("pbmc")
        content = None if found is None else found.read_bytes().decode()
        records = sorted(
            (r.created_at, bytes(Path(eng._stored_path(r)).read_bytes()))
            for r in eng._files().filter(key="data/pbmc.h5ad") if Path(eng._stored_path(r)).is_file()
        )
        saved.write_bytes(b"edited after the session")  # no record has these bytes
        emit(state=state, content=content,
             newest=records[-1][1].decode(), edited=list(eng.replay_reference_state("pbmc")))
    """)
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws"), "XYX": "1"})
    assert r["newest"] == "bytes Y"  # the newest record is not the file's
    assert (r["state"], r["content"]) == ("ok", "session bytes")
    assert r["edited"] == [None, "changed"]


def test_a_later_catalogued_save_leaves_the_sessions_one_record_as_the_reference(
    tmp_path,
):
    code = textwrap.dedent(REPLAY) + textwrap.dedent("""
        eng.begin_run(SessionInfo("s2", ws, "0.0.0"))
        target = eng.keep_path(saved)  # an earlier version is kept aside
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(saved, target)
        saved.write_bytes(b"later bytes")
        eng.register_artifact(saved, **kw("data/pbmc.h5ad"))
        eng.end_run("completed")
        replay()
        found, state = eng.replay_reference_state("pbmc")
        content = None if found is None else found.read_bytes().decode()
        saved.write_bytes(b"edited outside Lobster")  # no record has these bytes
        emit(state=state, content=content, edited=list(eng.replay_reference_state("pbmc")))
    """)
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert (r["state"], r["content"]) == ("ok", "session bytes")
    assert r["edited"] == [None, "changed"]


def test_records_that_disagree_and_no_file_to_settle_it_are_changed(tmp_path):
    code = textwrap.dedent(REPLAY) + textwrap.dedent("""
        replay()
        saved.unlink()
        emit(gone=list(eng.replay_reference_state("pbmc")))
    """)
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws"), "XYX": "1"})
    assert r["gone"] == [None, "changed"]


def test_the_newest_save_of_the_session_wins_across_both_names(tmp_path):
    code = textwrap.dedent(REPLAY) + textwrap.dedent("""
        replay()
        saved.write_bytes(b"edited")  # the older record, at data/pbmc.h5ad
        found, state = eng.replay_reference_state("pbmc")
        emit(state=state, name=None if found is None else str(found.relative_to(ws)))
    """)
    r = _run(
        code,
        tmp_path,
        env_extra={"WS": str(tmp_path / "ws"), "ROOT_PBMC": "1"},
    )
    assert (r["state"], r["name"]) == ("ok", "pbmc.h5ad")


def test_a_replay_does_not_end_a_session_that_is_open(tmp_path):
    code = textwrap.dedent(REPLAY) + textwrap.dedent("""
        eng.begin_run(SessionInfo("live", ws, "0.0.0"))
        started = replay()
        live = eng.ln.Run.filter(name="live").get()
        emit(started=started, open=eng._run is not None and eng._run.uid == live.uid,
             unfinished=live.finished_at is None,
             replays=eng.ln.Run.filter(name__startswith="replay:").count())
    """)
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert r["started"] is None and r["open"] and r["unfinished"]
    assert r["replays"] == 0


def test_a_notebook_copied_under_another_key_is_not_tied_to_the_exporting_session(
    tmp_path,
):
    # LaminDB keeps one Transform per source: the second export of the same bytes
    # (key m) is the first one's Transform (key n), whose session saved nothing for m
    code = textwrap.dedent(REPLAY) + textwrap.dedent("""
        out = {}
        for label, key in (("n", "notebooks/n.ipynb"), ("m", "notebooks/m.ipynb"),
                           ("copy", "notebooks/copy.ipynb")):
            started = replay(key=key)
            out[label] = started.session_uid
            run = eng.ln.Run.get(uid=started.uid)
            out[label + "_initiated"] = run.initiated_by_run_id
            out[label + "_notebook"] = run.extra_data["notebook"]
            eng.end_replay("completed", verdict="MATCH", report_path=None, outputs=[])
        emit(**out, session=session.uid, session_id=session.id,
             transforms=eng.ln.Transform.filter(kind="notebook").count())
    """)
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws"), "TWIN": "1"})
    assert r["transforms"] == 1
    assert r["n"] == r["session"] and r["n_initiated"] == r["session_id"]
    for label in ("m", "copy"):
        assert r[label] is None and r[label + "_initiated"] is None
        assert r[label + "_notebook"] == f"notebooks/{label}.ipynb"


def test_ending_a_replay_records_the_verdict_report_and_outputs(tmp_path):
    code = textwrap.dedent(REPLAY) + textwrap.dedent("""
        started = replay()
        folder = ws / "replays" / started.uid
        folder.mkdir(parents=True)
        report = folder / "replay_report.json"
        report.write_text('{"verdict": "MISMATCH"}\\n')
        new = folder / "out_pbmc.h5ad"
        new.write_bytes(b"different bytes")
        same = folder / "out_same.h5ad"
        same.write_bytes(b"session bytes")  # what the session saved
        eng.end_replay("completed", verdict="MISMATCH", report_path=report,
                       outputs=[new, folder / "gone.h5ad", same])
        run = eng.ln.Run.get(uid=started.uid)
        records = {r["key"]: r for r in eng.list_artifacts()}
        data = eng.ln.Artifact.filter(key="data/pbmc.h5ad")
        emit(extra=run.extra_data, finished=run.finished_at is not None,
             status=run._status_code, report=run.report.path.read_text(),
             report_kind=run.report.kind, report_file=run.report.path.name, report_listed=[k for k in records if "report" in k],
             keys=sorted(records), versions=data.count(),
             new_run=records[f"replays/{started.uid}/out_pbmc.h5ad"]["run"],
             recreated=data.get().recreating_runs.filter(id=run.id).exists(),
             uid=started.uid,
             session_files=eng.ln.Artifact.filter(key="data/pbmc.h5ad", is_latest=True).count(),
             closed=eng._run is None, again=eng.replay_reference("pbmc"))
    """)
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert r["extra"]["verdict"] == "MISMATCH" and r["extra"]["kind"] == "replay"
    assert r["finished"] and r["status"] == 0 and r["closed"]
    assert (
        r["report"] == '{"verdict": "MISMATCH"}\n'
        and r["report_kind"] == "__lamindb_run__"
    )
    assert r["report_file"].endswith(".json")  # a JSON report, not a step log
    assert r["report_listed"] == []  # the report is a run record, not one of the files
    # the output that differs is a record of its own under the replay's key; the one
    # the session already recorded is not recorded again
    assert r["new_run"] == "replay:n"
    assert [k for k in r["keys"] if k.startswith("replays/")] == [
        f"replays/{r['uid']}/out_pbmc.h5ad"
    ]
    assert r["recreated"]
    assert r["versions"] == 1 and r["session_files"] == 1  # no new version of user data
    assert r["net"] == []


def test_a_replay_output_with_the_bytes_of_a_changed_file_does_not_take_its_record(
    tmp_path,
):
    code = textwrap.dedent(REPLAY) + textwrap.dedent("""
        record = eng.ln.Artifact.get(key="data/pbmc.h5ad")
        before = str(eng._stored_path(record))
        saved.write_bytes(b"changed outside Lobster")  # the record's file is not intact
        started = replay()
        folder = ws / "replays" / started.uid
        folder.mkdir(parents=True)
        out = folder / "out_pbmc.h5ad"
        out.write_bytes(b"session bytes")  # the bytes the record holds
        eng.end_replay("completed", verdict="MATCH", report_path=None, outputs=[out])
        after = eng.ln.Artifact.get(uid=record.uid)
        emit(moved=str(eng._stored_path(after)) != before,
             records=eng.ln.Artifact.filter(key__startswith="replays/").count(),
             linked=after.recreating_runs.filter(uid=started.uid).exists())
    """)
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert r["moved"] is False and r["records"] == 0 and r["linked"] is True
    assert r["net"] == []


def test_a_replay_of_a_notebook_the_session_did_not_export_has_no_session(tmp_path):
    code = textwrap.dedent(REPLAY) + textwrap.dedent("""
        edited = NB.replace("[]", "[ ]")  # changed since the export
        started = replay(source=edited)
        run = eng.ln.Run.get(uid=started.uid)
        emit(session=started.session_uid, initiated=run.initiated_by_run_id,
             reference=eng.replay_reference("pbmc"),
             versions=sorted(t.version for t in eng.ln.Transform.filter(key="notebooks/n.ipynb")),
             owner=eng.ln.Transform.get(source_code=edited).run_id)
    """)
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert r["session"] is None and r["initiated"] is None and r["reference"] is None
    assert len(r["versions"]) == 2 and r["owner"] is None
    assert r["net"] == []


def test_a_notebook_that_looks_like_it_holds_a_secret_is_not_replayed_into_the_catalog(
    tmp_path,
):
    code = textwrap.dedent(REPLAY) + textwrap.dedent("""
        secret = NB.replace("[]", '["password = hunter2hunter2xx"]')
        started = replay(source=secret)
        emit(started=started, runs=eng.ln.Run.filter(name__startswith="replay:").count(),
             notebooks=eng.ln.Transform.filter(kind="notebook").count(), open=eng._run)
    """)
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert r["started"] is None and r["runs"] == 0 and r["open"] is None
    assert r["notebooks"] == 1  # the session's own export
    assert r["net"] == []


def test_a_replay_in_a_named_instance_is_recorded_the_same_way(tmp_path):
    inst = tmp_path / "inst"
    _run(
        f"""
        import lamindb_setup
        lamindb_setup.init(storage={str(inst)!r}, name="mbreplay")
        emit()
        """,
        tmp_path,
    )
    code = textwrap.dedent(REPLAY) + textwrap.dedent("""
        started = replay()
        found = eng.replay_reference("pbmc")
        folder = ws / "replays" / started.uid
        folder.mkdir(parents=True)
        report = folder / "replay_report.json"
        report.write_text('{"verdict": "MISMATCH"}\\n')
        new = folder / "out_pbmc.h5ad"
        new.write_bytes(b"different bytes")
        eng.end_replay("completed", verdict="MISMATCH", report_path=report, outputs=[new])
        run = eng.ln.Run.get(uid=started.uid)
        out = eng.ln.Artifact.get(key=f"replays/{started.uid}/out_pbmc.h5ad")
        emit(mode=eng.mode, found=None if found is None else found.read_bytes().decode(),
             in_storage=eng._in_storage(found),
             initiated=run.initiated_by_run_id is not None,
             report=run.report.path.read_text(),
             out=Path(str(out.path)).read_bytes().decode(), out_run=out.run.name,
             in_workspace=sorted(p.name for p in folder.iterdir()))
    """)
    r = _run(
        code,
        tmp_path,
        env_extra={
            "WS": str(tmp_path / "ws"),
            "INST_SLUG": "anonymous/mbreplay",
            "INST": str(inst),
        },
    )
    assert r["mode"] == "instance" and r["found"] == "session bytes" and r["in_storage"]
    assert r["initiated"] and r["report"] == '{"verdict": "MISMATCH"}\n'
    assert r["out"] == "different bytes" and r["out_run"] == "replay:n"
    assert r["in_workspace"] == ["out_pbmc.h5ad", "replay_report.json"]  # not moved
    assert r["net"] == []


COPY_RACE = """
    import hashlib
    from pathlib import Path
    from lobster.core.catalog import lamin as lamin_module
    from lobster.core.catalog.lamin import LaminCatalog
    from lobster.core.catalog.port import SessionInfo
    ws = Path(os.environ["WS"]).resolve()
    (ws / "data").mkdir(parents=True)
    x = ws / "data" / "x.csv"
    eng = LaminCatalog(ws, instance="anonymous/mbrace")
    eng.begin_run(SessionInfo("s1", ws, "0.0.0"))
    kw = dict(key="data/x.csv", kind="dataset", otype=None, description="d", meta={})
    sha = lambda data: hashlib.sha256(data).hexdigest()
    real_copy, copies = lamin_module._copy_file, []

    def racing(rewrites):
        # The file is rewritten after it was read and before it is copied.
        def copy(source, target, *args, **kwargs):
            copies.append(Path(target).name)
            if len(copies) <= rewrites:
                Path(source).write_bytes(b"rewritten %d" % len(copies))
            return real_copy(source, target, *args, **kwargs)
        return copy
"""


def _race_instance(tmp_path):
    inst = tmp_path / "inst"
    _run(
        f"""
        import lamindb_setup
        lamindb_setup.init(storage={str(inst)!r}, name="mbrace")
        emit()
        """,
        tmp_path,
    )
    return {"WS": str(tmp_path / "ws"), "INST": str(inst)}


def test_a_copy_of_a_file_rewritten_during_the_copy_is_recorded_as_what_it_holds(
    tmp_path,
):
    """A named instance hashes the file, then copies it: the record and the copy's name come
    from the bytes that were stored, so a rewrite in between cannot split them."""
    env = _race_instance(tmp_path)
    code = textwrap.dedent(COPY_RACE) + textwrap.dedent("""
        x.write_bytes(b"first bytes")
        lamin_module._copy_file = racing(1)
        r = eng.register_artifact(x, **kw)
        record = eng.ln.Artifact.get(r.uid)
        emit(stored=r.path.read_bytes().decode(),
             name=r.path.name,
             recorded=record.extra_data["lobster"]["sha256"],
             new=sha(b"rewritten 1"), old=sha(b"first bytes"),
             copies=len(copies), count=eng.ln.Artifact.filter(key="data/x.csv").count())
    """)
    r = _run(code, tmp_path, env_extra=env)
    assert r["stored"] == "rewritten 1"
    assert r["recorded"] == r["new"] != r["old"]
    assert r["name"] == r["new"] + ".csv"
    assert r["copies"] == 2 and r["count"] == 1  # the second copy is the one kept
    assert r["net"] == []


def test_a_file_that_keeps_changing_while_it_is_copied_is_not_recorded(tmp_path):
    env = _race_instance(tmp_path)
    code = textwrap.dedent(COPY_RACE) + textwrap.dedent("""
        x.write_bytes(b"first bytes")
        lamin_module._copy_file = racing(99)
        r = eng.register_artifact(x, **kw)
        eng.end_run("completed")
        emit(ref=r, copies=len(copies),
             count=eng.ln.Artifact.filter(key="data/x.csv").count(),
             left=sorted(p.name for p in (eng._storage_root / "lobster").rglob("*")
                         if p.is_file() and p.suffix in (".csv", ".partial")))
    """)
    r, out, err = _run(code, tmp_path, env_extra=env, streams=True)
    assert r["ref"] is None and r["copies"] == 2 and r["count"] == 0
    assert r["left"] == []  # no copy without a record
    assert "data/x.csv changed while it was being copied" in out + err
    assert r["net"] == []


def test_a_copy_that_cannot_be_read_back_leaves_no_file_behind(tmp_path):
    env = _race_instance(tmp_path)
    code = textwrap.dedent(COPY_RACE) + textwrap.dedent("""
        real_sha = lamin_module._sha256

        def unreadable_copy(p):
            if Path(p).name.startswith(".staged-"):
                raise OSError("disk error")
            return real_sha(p)

        lamin_module._sha256 = unreadable_copy
        x.write_bytes(b"first bytes")
        try:
            eng.register_artifact(x, **kw)
            raised = False
        except OSError:
            raised = True
        emit(raised=raised, count=eng.ln.Artifact.filter(key="data/x.csv").count(),
             left=sorted(p.name for p in (eng._storage_root / "lobster").rglob("*")
                         if p.is_file()))
    """)
    r = _run(code, tmp_path, env_extra=env)
    assert r["raised"] and r["count"] == 0
    assert r["left"] == []
    assert r["net"] == []


def test_a_copy_put_back_for_a_record_must_hold_the_recorded_bytes(tmp_path):
    env = _race_instance(tmp_path)
    code = textwrap.dedent(COPY_RACE) + textwrap.dedent("""
        x.write_bytes(b"first bytes")
        r1 = eng.register_artifact(x, **kw)
        r1.path.unlink()  # the copy left the instance's storage
        lamin_module._copy_file = racing(1)
        r2 = eng.register_artifact(x, **kw)  # the same bytes again, rewritten meanwhile
        emit(same=r2.uid == r1.uid, existing=r2.existing,
             stored=r2.path.read_bytes().decode(),
             recorded=eng.ln.Artifact.get(r2.uid).extra_data["lobster"]["sha256"],
             new=sha(b"rewritten 1"))
    """)
    r = _run(code, tmp_path, env_extra=env)
    assert r["stored"] == "rewritten 1" and r["recorded"] == r["new"]
    assert not r["same"] and not r["existing"]  # a record of the new bytes
    assert r["net"] == []


def test_a_copy_put_back_for_a_record_at_another_key_recreates_its_folder(tmp_path):
    env = _race_instance(tmp_path)
    code = textwrap.dedent(COPY_RACE) + textwrap.dedent("""
        import shutil
        x.write_bytes(b"first bytes")
        r1 = eng.register_artifact(x, **kw)
        shutil.rmtree(r1.path.parent)  # the copy and its folder left the storage
        r2 = eng.register_artifact(x, **dict(kw, key="data/y.csv"))
        emit(same=r2.uid == r1.uid, stored=r1.path.read_bytes().decode())
    """)
    r = _run(code, tmp_path, env_extra=env)
    assert r["same"] and r["stored"] == "first bytes"
    assert r["net"] == []


def test_a_save_the_catalog_did_not_prepare_is_repaired_by_the_next(tmp_path):
    """Caller level: a save during start-up overwrote a recorded version.

    The start-up window (no keep_path) is simulated by bypassing keep_path for
    one save. The overwritten record is marked superseded; saving its bytes
    again brings it back to the file name, with nothing left in data/.
    """
    code = textwrap.dedent(ADATA) + textwrap.dedent("""
        from pathlib import Path
        from lobster.core.catalog import factory
        from lobster.core.runtime.data_manager import DataManagerV2
        ws = Path(os.environ["WS"]).resolve()
        dm = DataManagerV2(workspace_path=ws)
        catalog = factory.begin_session_run(dm, ws, "session_a")
        dm.modalities["x"] = adata(1)
        v1 = Path(dm.save_modality("x", "x.h5ad"))
        v1_bytes = v1.read_bytes()
        catalog.keep_path = lambda p: None  # the start-up window
        dm.modalities["x"] = adata(2)
        dm.save_modality("x", "x.h5ad")
        del catalog.keep_path
        dm.modalities["x"] = adata(1)
        again = Path(dm.save_modality("x", "x.h5ad"))
        st = catalog.status()
        catalog.close("completed")
        emit(degraded=st.degraded, reason=st.reason, again=again.name,
             again_is_v1=(again.read_bytes() == v1_bytes),
             data_files=sorted(p.name for p in (ws / "data").iterdir()),
             kept=sorted(p.name for p in (ws / ".lamindb/versions").rglob("*.h5ad")
                         if p.is_file()))
    """)
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert not r["degraded"], r["reason"]
    assert r["again"] == "x.h5ad" and r["again_is_v1"]
    assert r["data_files"] == ["x.h5ad"] and r["kept"] == ["v0001.h5ad"]
    assert r["net"] == []


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root writes whatever the folder mode",
)
def test_a_version_that_cannot_be_kept_fails_open_and_is_marked(tmp_path):
    code = """
        from pathlib import Path
        from lobster.core.catalog.lamin import LaminCatalog
        from lobster.core.catalog.port import SessionInfo
        ws = Path(os.environ["WS"]).resolve()
        (ws / "data").mkdir(parents=True)
        x = ws / "data" / "x.csv"
        eng = LaminCatalog(ws)
        eng.begin_run(SessionInfo("s1", ws, "0.0.0"))
        kw = dict(key="data/x.csv", kind="dataset", otype=None, description="d", meta={})
        x.write_text("x\\n1\\n")
        r1 = eng.register_artifact(x, **kw)
        store = ws / ".lamindb" / "versions"
        store.mkdir(parents=True)
        store.chmod(0o500)
        try:
            target = eng.keep_path(x)
            try:  # what the saving thread does; it fails here
                target.parent.mkdir(parents=True)
                shutil.copy2(x, target)
                refused = None
            except OSError as exc:
                refused = type(exc).__name__
            x.write_text("x\\n2\\n")
            eng.register_artifact(x, **kw)
        finally:
            store.chmod(0o700)
        tag = eng.ln.Artifact.get(r1.uid).extra_data["lobster"]
        latest = [a.version for a in eng.ln.Artifact.filter(key="data/x.csv",
                                                             is_latest=True)]
        emit(target=target.name, refused=refused, tag=tag, latest=latest)
    """
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert r["target"] == "v0000.csv" and r["refused"] == "PermissionError"
    assert r["tag"]["not_kept"] == "overwritten before it could be kept"
    assert r["tag"]["superseded_by"]
    assert r["latest"] == ["0001"]
    assert r["net"] == []


def test_a_record_back_at_its_name_is_the_one_latest(tmp_path):
    """Saves the guard did not see (A, B, A): the name holds A, so A is latest.

    A new save C after that is then the one latest.
    """
    code = """
        from pathlib import Path
        from lobster.core.catalog.lamin import LaminCatalog, _lamin_hash
        from lobster.core.catalog.port import SessionInfo
        ws = Path(os.environ["WS"]).resolve()
        (ws / "data").mkdir(parents=True)
        x = ws / "data" / "x.csv"
        eng = LaminCatalog(ws)
        eng.begin_run(SessionInfo("s1", ws, "0.0.0"))
        kw = dict(key="data/x.csv", kind="dataset", otype=None, description="d", meta={})
        def rows():
            out = []
            for a in eng.ln.Artifact.filter(key="data/x.csv").order_by("created_at"):
                stored = eng._stored_path(a)
                lob = (a.extra_data or {}).get("lobster") or {}
                out.append([a.version, a.is_latest, stored.is_file()
                            and _lamin_hash(stored) == a.hash, "superseded_by" in lob])
            return out
        for text in ("a\\n1\\n", "b\\n2\\n", "a\\n1\\n"):
            x.write_text(text)
            eng.register_artifact(x, **kw)
        before = rows()
        kept = eng.keep_path(x)  # as the saving thread does before it writes
        kept.parent.mkdir(parents=True, exist_ok=True)
        os.link(x, kept)
        x.unlink()
        x.write_text("c\\n3\\n")
        eng.register_artifact(x, **kw)
        emit(before=before, after=rows(), name=x.read_text())
    """
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    # version, latest, holds its bytes, marked
    assert r["before"] == [["0000", True, True, False], ["0001", False, False, True]]
    latest = [row[0] for row in r["after"] if row[1]]
    assert latest == [r["after"][-1][0]] and r["after"][-1][0] not in ("0000", "0001")
    assert r["after"][-1][2] and r["name"] == "c\n3\n"
    assert r["after"][0][2], r  # A's bytes are kept when C overwrites them
    assert r["net"] == []


def test_a_relocated_record_at_its_name_becomes_the_latest(tmp_path):
    code = """
        from pathlib import Path
        from lobster.core.catalog.lamin import LaminCatalog
        from lobster.core.catalog.port import SessionInfo
        ws = Path(os.environ["WS"]).resolve()
        (ws / "data").mkdir(parents=True)
        x = ws / "data" / "x.csv"
        eng = LaminCatalog(ws)
        eng.begin_run(SessionInfo("s1", ws, "0.0.0"))
        kw = dict(key="data/x.csv", kind="dataset", otype=None, description="d", meta={})
        x.write_text("a\\n1\\n")
        first = eng.register_artifact(x, **kw)
        x.write_text("b\\n2\\n")
        eng.register_artifact(x, **kw)
        x.write_text("a\\n1\\n")
        record = eng.ln.Artifact.get(first.uid)
        eng._relocate(record, x)
        record.save()
        latest = [a.version for a in eng.ln.Artifact.filter(key="data/x.csv",
                                                             is_latest=True)]
        emit(latest=latest)
    """
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert r["latest"] == ["0000"]
    assert r["net"] == []


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root reads files whatever their mode",
)
def test_unreadable_files_fail_open_and_are_tagged(tmp_path):
    code = """
        from pathlib import Path
        import shutil
        from lobster.core.catalog.lamin import LaminCatalog
        from lobster.core.catalog.port import SessionInfo
        ws = Path(os.environ["WS"]).resolve()
        (ws / "data").mkdir(parents=True)
        eng = LaminCatalog(ws)
        eng.begin_run(SessionInfo("s1", ws, "0.0.0"))
        kw = dict(kind="dataset", otype=None, description="d", meta={})
        # 1. keep_path cannot read the recorded file: the save goes ahead, tagged
        a = ws / "data" / "a.csv"
        a.write_text("a\\n1\\n")
        ra = eng.register_artifact(a, key="data/a.csv", **kw)
        a.chmod(0)
        guard = eng.keep_path(a)
        a.chmod(0o644)
        a.write_text("a\\n2\\n")
        eng.register_artifact(a, key="data/a.csv", **kw)  # saves the pending tag
        tag_a = eng.ln.Artifact.get(ra.uid).extra_data["lobster"]
        # 2. an identical re-save cannot read the recorded file: keep the new one
        b = ws / "data" / "b.csv"
        b.write_text("b\\n1\\n")
        rb = eng.register_artifact(b, key="data/b.csv", **kw)
        b2 = ws / "data" / "b__v2.csv"
        shutil.copy(b, b2)
        b.chmod(0)
        again = eng.register_artifact(b2, key="data/b.csv", **kw)
        b.chmod(0o644)
        rec_b = eng.ln.Artifact.get(rb.uid)
        emit(guard=guard, tag_a=tag_a, again_existing=again.existing,
             again_uid=again.uid, rb_uid=rb.uid, b_path=Path(str(rec_b.path)).name,
             tag_b=rec_b.extra_data["lobster"])
    """
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert r["guard"] is None
    assert r["tag_a"]["unreadable"] == "data/a.csv at write"
    assert r["again_uid"] == r["rb_uid"] and not r["again_existing"]
    assert r["b_path"] == "b__v2.csv"
    assert r["tag_b"]["unreadable"] == "data/b.csv at re-save"
    assert r["tag_b"]["relocated_from"] == "data/b.csv"
    assert r["net"] == []


def test_unfinished_record_is_not_reported_as_existing(tmp_path):
    """A record whose earlier save never completed (LaminDB's storage flag)."""
    code = """
        from pathlib import Path
        import shutil
        from lobster.core.catalog.lamin import LaminCatalog
        from lobster.core.catalog.port import SessionInfo
        ws = Path(os.environ["WS"]).resolve()
        (ws / "data").mkdir(parents=True)
        eng = LaminCatalog(ws)
        eng.begin_run(SessionInfo("s1", ws, "0.0.0"))
        kw = dict(key="data/x.csv", kind="dataset", otype=None, description="d", meta={})
        x = ws / "data" / "x.csv"
        x.write_text("x\\n1\\n")
        r1 = eng.register_artifact(x, **kw)
        rec = eng.ln.Artifact.get(r1.uid)
        rec._storage_ongoing = True
        rec.save()
        same = eng.register_artifact(x, **kw)
        rec = eng.ln.Artifact.get(r1.uid)
        rec._storage_ongoing = True
        rec.save()
        y = ws / "data" / "x__v2.csv"
        shutil.copy(x, y)
        other = eng.register_artifact(y, **kw)
        rec = eng.ln.Artifact.get(r1.uid)
        emit(same_existing=same.existing, other_existing=other.existing,
             other_uid=other.uid, r1_uid=r1.uid, path=Path(str(rec.path)).name,
             ongoing=rec._storage_ongoing)
    """
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert not r["same_existing"] and not r["other_existing"]
    assert r["other_uid"] == r["r1_uid"] and r["path"] == "x__v2.csv"
    assert r["ongoing"] is False
    assert r["net"] == []


def test_a_notebook_exported_again_is_a_new_version_of_the_same_transform(tmp_path):
    """Pins LaminDB 2.12.2: the same key with other source is a new version, not an error.

    The same bytes again add nothing; the same bytes under another key come back as
    the first key's transform (LaminDB keeps the original key).
    """
    code = textwrap.dedent("""
        from pathlib import Path
        from unittest.mock import Mock
        from lobster.core.catalog import factory
        from lobster.core.runtime.data_manager import DataManagerV2
        ws = Path(os.environ["WS"]).resolve()
        dm = DataManagerV2(workspace_path=ws)
        catalog = factory.begin_session_run(dm, ws, "session_a")
        nb = ws / "notebooks" / "session.ipynb"
        nb.parent.mkdir(parents=True, exist_ok=True)
        dm._notebook_exporter = Mock(export=Mock(return_value=nb))
        dm.log_tool_usage("qc", {}, description="d")
        for text in ('{"cells": ["one"]}', '{"cells": ["one"]}', '{"cells": ["one", "two"]}'):
            nb.write_text(text)
            dm.export_notebook("session", description="exported session")
        other = catalog.register_transform(
            key="notebooks/other.ipynb", kind="notebook",
            source='{"cells": ["one", "two"]}', description="copy",
        )
        catalog.close("completed")
        emit(other=[other.uid, other.key])
        """)
    readback = """
        from pathlib import Path
        ws = Path(os.environ["WS"])
        os.environ["LAMIN_SETTINGS_DIR"] = str(ws / ".lamindb" / "settings")
        import lamindb_setup as lns
        lns.connect(json.loads((ws / ".lamindb/lobster.json").read_text())["slug"],
                    _write_settings=False)
        import lamindb as ln
        emit(transforms=[[t.key, t.uid, t.is_latest] for t in
                         ln.Transform.filter(kind="notebook").order_by("created_at")])
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    r = _run(code, tmp_path, env_extra={"WS": str(ws)})
    back = _run(readback, tmp_path, env_extra={"WS": str(ws)})
    (key1, uid1, latest1), (key2, uid2, latest2) = back["transforms"]
    assert key1 == key2 == "notebooks/session.ipynb"  # three exports, two versions
    assert uid1[:-4] == uid2[:-4]  # one family: only the version suffix moves
    assert [uid1[-4:], uid2[-4:]] == ["0000", "0001"]
    assert [latest1, latest2] == [False, True]
    assert r["other"] == [uid2, key2]
    assert r["net"] == back["net"] == []


def test_a_closed_stderr_does_not_cost_a_save_its_record(tmp_path):
    """LaminDB's log call must not raise into the save when sys.stderr was closed."""
    code = textwrap.dedent(ADATA) + textwrap.dedent("""
        from pathlib import Path
        from lobster.core.catalog import factory
        from lobster.core.runtime.data_manager import DataManagerV2
        ws = Path(os.environ["WS"]).resolve()
        dm = DataManagerV2(workspace_path=ws)
        catalog = factory.begin_session_run(dm, ws, "session_a")
        dm.modalities["x"] = adata(1)
        dm.save_modality("x", "x.h5ad")
        closed = open(os.devnull, "w")
        closed.close()
        real_stderr, sys.stderr = sys.stderr, closed
        try:
            for seed in (2, 3):
                dm.modalities["x"] = adata(seed)
                dm.save_modality("x", "x.h5ad")
        finally:
            sys.stderr = real_stderr
        catalog.close("completed")
        emit(type=type(catalog).__name__)
        """)
    readback = """
        from pathlib import Path
        ws = Path(os.environ["WS"])
        os.environ["LAMIN_SETTINGS_DIR"] = str(ws / ".lamindb" / "settings")
        import lamindb_setup as lns
        lns.connect(json.loads((ws / ".lamindb/lobster.json").read_text())["slug"],
                    _write_settings=False)
        import lamindb as ln
        emit(versions=[a.version for a in
                       ln.Artifact.filter(key="data/x.h5ad").order_by("created_at")])
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    _run(code, tmp_path, env_extra={"WS": str(ws)})
    back = _run(readback, tmp_path, env_extra={"WS": str(ws)})
    assert back["versions"] == ["0000", "0001", "0002"]


def test_a_one_shot_exit_waits_for_a_slow_engine_start(tmp_path):
    """The process ends while LaminDB is still starting; its run is recorded anyway."""
    code = textwrap.dedent("""
        import time
        from pathlib import Path
        from lobster.core.catalog import factory, lamin, thread
        from lobster.core.runtime.data_manager import DataManagerV2
        # A short start-up wait and a slow start stand in for a loaded machine.
        thread.ThreadedCatalog.__init__.__kwdefaults__["startup_wait"] = 0.5
        original = lamin.LaminCatalog.__init__
        def slow(self, *args, **kwargs):
            time.sleep(1.5)
            original(self, *args, **kwargs)
        lamin.LaminCatalog.__init__ = slow
        ws = Path(os.environ["WS"]).resolve()
        dm = DataManagerV2(workspace_path=ws)
        catalog = factory.begin_session_run(dm, ws, "session_a")
        factory.note_run_status(dm, "completed")
        emit(type=type(catalog).__name__)
        """)
    readback = """
        from pathlib import Path
        ws = Path(os.environ["WS"])
        config = ws / ".lamindb" / "lobster.json"
        runs = []
        if config.is_file():
            os.environ["LAMIN_SETTINGS_DIR"] = str(ws / ".lamindb" / "settings")
            import lamindb_setup as lns
            lns.connect(json.loads(config.read_text())["slug"], _write_settings=False)
            import lamindb as ln
            runs = [[r.name, r.status] for r in ln.Run.filter(name="session_a")]
        emit(runs=runs)
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    first = _run(code, tmp_path, env_extra={"WS": str(ws)})
    assert first["type"] == "ThreadedCatalog" and first["net"] == []
    back = _run(readback, tmp_path, env_extra={"WS": str(ws)})
    assert back["runs"] == [["session_a", "completed"]]


def test_stdout_clean_with_catalog(tmp_path):
    """LaminDB logs to stdout by default; a ``--json`` command must stay parseable."""
    ws = tmp_path / "ws_cmd"
    ws.mkdir()
    env = _isolated_env(tmp_path, {}, False)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from lobster.cli import app; "
            "sys.argv = ['lobster', 'command', 'data', '--json', '--workspace', "
            f"{str(ws)!r}]; app()",
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    json.loads(proc.stdout)  # the whole of stdout is one JSON document
    assert "initialized lamindb" in proc.stderr  # the catalog was created, quietly


def test_a_recorded_session_leaves_stdout_to_the_program(tmp_path):
    """The message for a new file version goes to stderr, not into the program's stdout."""
    code = textwrap.dedent(ADATA) + textwrap.dedent("""
        from pathlib import Path
        from lobster.core.catalog import factory
        from lobster.core.runtime.data_manager import DataManagerV2
        ws = Path(os.environ["WS"])
        dm = DataManagerV2(workspace_path=ws)
        catalog = factory.begin_session_run(dm, ws, "session_a")
        for seed in (1, 2):
            dm.modalities["x"] = adata(seed)
            dm.save_modality("x", "x.h5ad")
        catalog.close("completed")
        emit(type=type(catalog).__name__)
        """)
    ws = tmp_path / "ws"
    ws.mkdir()
    result, out, err = _run(code, tmp_path, env_extra={"WS": str(ws)}, streams=True)
    assert result["type"] == "ThreadedCatalog"
    assert [line for line in out.splitlines() if not line.startswith("RESULT ")] == []
    assert "creating new artifact version" in err


def test_lamin_commands_on_a_real_catalog(tmp_path):
    code = textwrap.dedent(ADATA) + textwrap.dedent("""
        from pathlib import Path
        from typer.testing import CliRunner
        from lobster.cli_internal.commands.light.lamin_commands import lamin_app
        from lobster.core.catalog import factory
        from lobster.core.runtime.data_manager import DataManagerV2
        ws = Path(os.environ["WS"])
        cli = CliRunner()
        def run(*args):
            result = cli.invoke(lamin_app, [*args, "-w", str(ws)])
            assert result.exit_code == 0, result.output
            return result.output
        before = json.loads(run("status", "--json"))
        init_out = run("init")
        dm = DataManagerV2(workspace_path=ws)
        catalog = factory.begin_session_run(dm, ws, "session_a")
        for seed in (1, 2):
            dm.modalities["x"] = adata(seed)
            dm.save_modality("x", "x.h5ad")
        catalog.close("completed")
        emit(before=before, init_out=init_out, status=json.loads(run("status", "--json")),
             status_text=run("status"), files=json.loads(run("ls", "--json")),
             runs=json.loads(run("ls", "--runs", "--json")), ls_text=run("ls"))
        """)
    ws = tmp_path / "ws"
    ws.mkdir()
    r = _run(code, tmp_path, env_extra={"WS": str(ws)}, drop_enable=True)
    assert r["before"]["enabled"] is False and r["before"]["slug"] is None
    assert "Workspace catalog ready: anonymous/lobster-ws-" in r["init_out"]
    status = r["status"]
    assert status["enabled"] is True and status["enabled_source"] == "workspace"
    assert status["contents"]["artifacts"] == 2 and status["contents"]["runs"] == 1
    assert status["contents"]["last_run"]["name"] == "session_a"
    assert status["contents"]["last_run"]["status"] == "completed"
    assert "Files: 2, sessions: 1, notebooks: 0" in r["status_text"]
    assert "Last session: session_a (completed, started " in r["status_text"]
    assert [(f["key"], f["latest"], f["file"]) for f in r["files"]] == [
        ("data/x.h5ad", True, "data/x.h5ad"),
        ("data/x.h5ad", False, ".lamindb/versions/data/x.h5ad/v0000.h5ad"),
    ]
    assert [(x["name"], x["outputs"]) for x in r["runs"]] == [("session_a", 2)]
    assert "file: .lamindb/versions/data/x.h5ad/v0000.h5ad" in r["ls_text"]
    assert r["net"] == []


def test_ls_runs_marks_what_each_run_recorded(tmp_path):
    code = textwrap.dedent("""
        from pathlib import Path
        from typer.testing import CliRunner
        from lobster.cli_internal.commands.light.lamin_commands import lamin_app
        from lobster.core.catalog import lamin
        from lobster.core.catalog.port import SessionInfo
        ws = Path(os.environ["WS"]).resolve()
        log = ws / "step_log.jsonl"
        log.write_text('{"step": 1}\\n')
        engine = lamin.LaminCatalog(ws)
        def record(name, report):
            engine.begin_run(SessionInfo(name, ws, "0.0.0"))
            engine.end_run("completed", report)
        record("with_both", log)
        record("env_only", None)
        engine._environment = lambda version: None  # a capture that gave nothing
        record("neither", None)
        cli = CliRunner()
        def run(*args):
            result = cli.invoke(lamin_app, [*args, "-w", str(ws)])
            assert result.exit_code == 0, result.output
            return result.output
        emit(runs=json.loads(run("ls", "--runs", "--json")), text=run("ls", "--runs"),
             status=json.loads(run("status", "--json")))
        """)
    ws = tmp_path / "ws"
    ws.mkdir()
    r = _run(code, tmp_path, env_extra={"WS": str(ws)})
    assert [(x["name"], x["report"], x["environment"]) for x in r["runs"]] == [
        ("neither", False, False),
        ("env_only", False, True),
        ("with_both", True, True),
    ]
    lines = r["text"].splitlines()
    assert lines[0].endswith("neither  (0 out, 0 in)")
    assert lines[1].endswith("env_only  (0 out, 0 in)  [environment]")
    assert lines[2].endswith("with_both  (0 out, 0 in)  [report, environment]")
    last = r["status"]["contents"]["last_run"]
    assert (last["name"], last["report"], last["environment"]) == (
        "neither",
        False,
        False,
    )
    assert r["status"]["contents"]["artifacts"] == 0  # the records are not files
    assert r["net"] == []


def test_missing_settings_give_a_hint_not_a_server_call(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    first = """
        from pathlib import Path
        from lobster.core.catalog.lamin import LaminCatalog
        LaminCatalog(Path(os.environ["WS"]))
        emit(ok=True)
    """
    _run(first, tmp_path, env_extra={"WS": str(ws)})
    import shutil

    shutil.rmtree(ws / ".lamindb" / "settings")
    second = """
        from pathlib import Path
        from lobster.core.catalog import factory
        cat = factory.get_catalog(Path(os.environ["WS"]))
        emit(reason=cat.status().reason)
    """
    r = _run(second, tmp_path, env_extra={"WS": str(ws)})
    assert (
        "settings are missing" in r["reason"] and "move .lamindb aside" in r["reason"]
    )
    assert r["net"] == []


def test_named_instance_through_the_factory_is_recorded_offline(tmp_path):
    """A named instance on local storage, created by the user with their own settings.

    Connecting works offline, and Lobster copies the file into the instance's
    storage itself and registers it there, so LaminDB never asks Lamin's servers
    (it would for a file outside the instance's storage).
    """
    setup = """
        import lamindb_setup as lns
        lns.init(storage=os.environ["STORE"], name="shared")
        emit(slug=lns.settings.instance.slug)
    """
    store = tmp_path / "store"
    store.mkdir()
    created = _run(setup, tmp_path, env_extra={"STORE": str(store)})
    assert created["net"] == []
    code = """
        from pathlib import Path
        from lobster.core.catalog import factory
        from lobster.core.catalog.port import SessionInfo
        ws = Path(os.environ["WS"]).resolve()
        (ws / "data").mkdir(parents=True)
        f = ws / "data" / "a.csv"
        f.write_text("x\\n1\\n")
        cat = factory.get_catalog(ws)
        cat.begin_run(SessionInfo("s1", ws, "0.0.0"))
        ref = cat.register_artifact(f, key="data/a.csv", kind="dataset", otype=None,
                                    description="d", meta={})
        st = cat.status()
        cat.close("completed")
        emit(mode=st.mode, instance=st.instance, reason=st.reason,
             ref=None if ref is None else [ref.key, ref.version, ref.existing],
             guard=cat.keep_path(f), ws_has_lamindb=(ws / ".lamindb").exists(),
             source_intact=(f.read_text() == "x\\n1\\n"),
             stored=sorted(p.name for p in Path(os.environ["STORE"]).rglob("*.csv")
                           if p.is_file()))
    """
    ws = tmp_path / "ws"
    r = _run(
        code,
        tmp_path,
        env_extra={
            "WS": str(ws),
            "STORE": str(store),
            "LOBSTER_LAMIN_INSTANCE": created["slug"],
        },
    )
    assert r["mode"] == "instance" and r["instance"] == created["slug"], r["reason"]
    assert r["ref"] == ["data/a.csv", "0000", False] and len(r["stored"]) == 1
    assert r["net"] == []
    assert r["source_intact"] and not r["ws_has_lamindb"]
    assert r["guard"] is None


def test_workspace_mode_refuses_after_foreign_lamin_import(tmp_path):
    code = """
        from pathlib import Path
        import lamindb_setup  # imported first, with the user's own settings
        from lobster.core.catalog import factory
        cat = factory.get_catalog(Path(os.environ["WS"]))
        emit(type=type(cat).__name__, reason=cat.status().reason)
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    r = _run(code, tmp_path, env_extra={"WS": str(ws)})
    assert r["type"] == "NullCatalog" and "already loaded" in r["reason"]
    assert not (ws / ".lamindb").exists()


def test_private_lamin_surfaces_still_exist(tmp_path):
    """Pins the private LaminDB API the engine depends on.

    This checks names and signatures only. A change in what these surfaces do is
    caught by the behavioural scenarios above (home and current directory left
    unchanged, records readable from a new process), not here.
    """
    code = """
        import inspect
        os.environ["LAMIN_SETTINGS_DIR"] = os.path.join(os.getcwd(), "s")
        import lamindb_setup
        from lamindb_setup import _init_instance, _connect_instance
        from lamindb_setup.core import _settings_store
        from lamindb_setup.core._settings_instance import InstanceSettings
        from lamindb.models import Artifact, Run
        init_src = inspect.getsource(_init_instance.init)
        persist = inspect.signature(InstanceSettings._persist).parameters
        settings_file = inspect.signature(_settings_store.instance_settings_file)
        from lamindb.core.storage import paths as storage_paths
        from lamindb_setup.core.hashing import hash_small_bytes
        import typing
        from lamindb.base.types import ArtifactKind
        fields = {f.name for f in Artifact._meta.get_fields()} | {f.name for f in Run._meta.get_fields()}
        emit(
            init_write_settings='kwargs.get("_write_settings", True)' in init_src,
            connect_write_settings="_write_settings" in inspect.getsource(_connect_instance.connect),
            persist_params=sorted(persist),
            settings_dir_env="LAMIN_SETTINGS_DIR" in inspect.getsource(_settings_store),
            fields=sorted(fields & {"_real_key", "_key_is_virtual", "_status_code", "extra_data",
                                    "environment", "report"}),
            small_hash=hash_small_bytes(b"abc"),
            run_kind="__lamindb_run__" in typing.get_args(ArtifactKind),
            completed_code=Run._meta.get_field("_status_code") is not None,
            settings_file_params=list(settings_file.parameters),
            storage_key=callable(getattr(storage_paths, "auto_storage_key_from_artifact", None)),
            storage_ongoing=isinstance(getattr(Artifact, "_storage_ongoing", None), property),
        )
    """
    r = _run(code, tmp_path)
    assert (
        r["init_write_settings"]
        and r["connect_write_settings"]
        and r["settings_dir_env"]
    )
    assert {"write_to_disk", "write_current_instance_file"} <= set(r["persist_params"])
    assert r["fields"] == [
        "_key_is_virtual",
        "_real_key",
        "_status_code",
        "environment",
        "extra_data",
        "report",
    ]
    assert r["small_hash"] == "kAFQmDzST7DWlj99KOF_cg"  # md5, url-safe base64
    assert r["run_kind"]  # the hidden artifact kind of LaminDB's own run records
    assert r["settings_file_params"] == ["name", "owner"]
    assert r["storage_key"] and r["storage_ongoing"]


def test_an_exported_lamin_api_key_does_not_log_in(tmp_path):
    """The workspace catalog stays anonymous and offline with LAMIN_API_KEY set."""
    code = """
        from pathlib import Path
        from lobster.core.catalog.lamin import LaminCatalog
        from lobster.core.catalog.port import SessionInfo
        ws = Path(os.environ["WS"]).resolve()
        (ws / "data").mkdir(parents=True)
        f = ws / "data" / "a.csv"
        f.write_text("a\\n1\\n")
        eng = LaminCatalog(ws)
        eng.begin_run(SessionInfo("s1", ws, "0.0.0"))
        ref = eng.register_artifact(f, key="data/a.csv", kind="dataset", otype=None,
                                    description="d", meta={})
        eng.end_run("completed")
        emit(slug=eng.slug, handle=eng._setup.settings.user.handle, ref=bool(ref),
             key_kept=os.environ.get("LAMIN_API_KEY"))
    """
    r = _run(
        code,
        tmp_path,
        env_extra={"WS": str(tmp_path / "ws"), "LAMIN_API_KEY": "not-a-real-key"},
    )
    assert r["slug"].startswith("anonymous/") and r["handle"] == "anonymous"
    assert r["ref"] and r["key_kept"] == "not-a-real-key"
    assert r["net"] == []


def test_a_named_local_instance_gets_copies_offline(tmp_path):
    """A named instance: saved bytes are copied into the instance's storage, offline."""
    inst = tmp_path / "inst"
    user = _run(
        f"""
        import hashlib
        from pathlib import Path
        import lamindb_setup
        lamindb_setup.init(storage={str(inst)!r}, name="mbtest")
        home = Path(os.environ["HOME"]) / ".lamin"
        emit(slug=lamindb_setup.settings.instance.slug, settings={{
            str(p.relative_to(home)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(home.rglob("*")) if p.is_file()}})
        """,
        tmp_path,
    )
    assert user["slug"] == "anonymous/mbtest" and user["net"] == []
    code = """
        import hashlib
        from pathlib import Path
        import lobster.core.catalog.lamin as lamin_mod
        from lobster.core.catalog.lamin import LaminCatalog, _lamin_hash, _sha256
        from lobster.core.catalog.port import SessionInfo
        ws = Path(os.environ["WS"]).resolve()
        inst = Path(os.environ["INST"]).resolve()
        (ws / "data").mkdir(parents=True)
        x = ws / "data" / "x.csv"
        eng = LaminCatalog(ws, instance="anonymous/mbtest")
        eng.begin_run(SessionInfo("s1", ws, "0.0.0"))
        run_uid = eng._run.uid
        log = inst.parent / "provenance.jsonl"
        log.write_text('{"step": 1}\\n')
        leaked = inst.parent / "leaked.jsonl"
        leaked.write_text('{"p": "export password=hunter2hunter2"}\\n')
        kw = dict(key="data/x.csv", kind="dataset", otype=None, description="d", meta={})
        refs = []
        for text in ("x\\n1\\n", "x\\n2\\n", "x\\n2\\n", "x\\n3\\n"):
            x.write_text(text)
            r = eng.register_artifact(x, **kw)
            refs.append([r.version, r.existing, str(r.path.relative_to(inst)),
                         _sha256(x)])
        x.write_text("x\\n2\\n")
        linked = eng.link_input(x)
        # A recorded copy that left the instance's storage is put back.
        x.write_text("x\\n1\\n")
        first = eng.ln.Artifact.filter(key="data/x.csv").order_by("created_at").first()
        Path(str(first.path)).unlink()
        again = eng.register_artifact(x, **kw)
        recopied = [again.version, again.existing, again.path.read_text()]
        eng.end_run("completed", report_path=log)
        report = eng.ln.Run.get(uid=run_uid).report
        report_row = [report.kind, str(Path(str(report.path)).relative_to(inst)),
                      Path(str(report.path)).read_text()]
        eng.begin_run(SessionInfo("s2", ws, "0.0.0"))
        run2 = eng._run.uid
        eng.end_run("completed", report_path=leaked)  # looks like a secret: no copy
        no_report = eng.ln.Run.get(uid=run2).report_id
        rows = [
            [a.version, a.is_latest, Path(str(a.path)).read_text(),
             a.hash == _lamin_hash(Path(str(a.path)))]
            for a in eng.ln.Artifact.filter(key="data/x.csv").order_by("created_at")
        ]
        home = Path(os.environ["HOME"]) / ".lamin"
        emit(slug=eng.slug, mode=eng.mode, refs=refs, rows=rows, recopied=recopied,
             linked=linked.version, ws_tree=sorted(
                 str(p.relative_to(ws)) for p in ws.rglob("*") if p.is_file()),
             copies=sorted(str(p.relative_to(inst)) for p in
                           (inst / "lobster").rglob("*") if p.is_file()),
             env=[a.hash for a in eng.ln.Artifact.filter(
                 kind="__lamindb_run__", description="environment")],
             report=report_row, no_report=no_report, run_uid=run_uid,
             settings={str(p.relative_to(home)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(home.rglob("*")) if p.is_file()})
    """
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws"), "INST": str(inst)})
    assert (r["slug"], r["mode"]) == ("anonymous/mbtest", "instance")
    assert [ref[:2] for ref in r["refs"]] == [
        ["0000", False],
        ["0001", False],
        ["0001", True],
        ["0002", False],
    ]
    for _, _, stored, digest in r[
        "refs"
    ]:  # <root>/lobster/<ws id>/<key>/<sha256><suffix>
        assert re.fullmatch(
            r"lobster/lobster-ws-[0-9a-f]{6}/data/x\.csv/[0-9a-f]{64}\.csv", stored
        )
        assert stored.endswith(f"/data/x.csv/{digest}.csv")
    assert r["rows"] == [
        ["0000", False, "x\n1\n", True],
        ["0001", False, "x\n2\n", True],
        ["0002", True, "x\n3\n", True],
    ]
    assert r["recopied"] == ["0000", True, "x\n1\n"]
    data = [c for c in r["copies"] if "/data/x.csv/" in c]
    assert len(data) == 3  # one per content, no orphan for the re-save
    # The run's environment: one copy in the instance's storage, none in the workspace.
    assert len(r["env"]) == 1
    # Decision 11 Nov 2026: the step log is copied too (after the scan), the one
    # that looks like it holds a secret is not.
    ws_id = Path(r["copies"][0]).parts[1]
    report_file = f"lobster/{ws_id}/reports/{r['run_uid']}.jsonl"
    assert [c for c in r["copies"] if c not in data] == [
        f"lobster/{ws_id}/environments/{r['env'][0]}.txt",
        report_file,
    ]
    assert r["report"] == ["__lamindb_run__", report_file, '{"step": 1}\n']
    assert r["no_report"] is None
    assert r["linked"] == "0001"
    assert r["ws_tree"] == ["data/x.csv"]  # the workspace file only, unchanged
    assert r["settings"] == user["settings"]  # the user's Lamin settings untouched
    assert r["net"] == []


ENVIRONMENT = """
    from pathlib import Path
    import typing
    import lobster.core.catalog.lamin as lamin_mod
    from lobster.core.catalog.lamin import LaminCatalog
    from lobster.core.catalog.port import SessionInfo
    ws = Path(os.environ["WS"]).resolve()
    ws.mkdir(exist_ok=True)
    scans = []

    def scan(lobster_version):  # one environment per Lobster version
        scans.append(lobster_version)
        return {"0.0.0": "# one\\na==1\\n", "0.0.1": "# two\\na==2\\n"}[lobster_version]

    lamin_mod.environment_text = scan
    eng = LaminCatalog(ws)
    runs = []
    for sid, version in (("s1", "0.0.0"), ("s2", "0.0.0"), ("s3", "0.0.1")):
        eng.begin_run(SessionInfo(sid, ws, version))
        runs.append(eng._run.uid)
        if sid != "s3":
            eng.end_run("completed")
    data = ws / "data" / "a.csv"
    data.parent.mkdir()
    data.write_text("x\\n1\\n")
    eng.register_artifact(
        data, key="data/a.csv", kind="dataset", otype=None, description="d", meta={}
    )
    other = ws / "data" / "b.csv"
    other.write_text("y\\n2\\n")
    eng.register_artifact(  # no kind: still listed
        other, key="data/b.csv", kind=None, otype=None, description="d", meta={}
    )
    eng.end_run("completed")
    ln = eng.ln
    envs = [[r.name, r.environment.uid, r.environment.kind, r.environment.hash,
             Path(str(r.environment.path)).read_text()]
            for r in (ln.Run.get(uid=u) for u in runs)]
    ws_tree = sorted(str(p.relative_to(ws)) for p in ws.rglob("*") if p.is_file()
                     and not str(p.relative_to(ws)).startswith(".lamindb"))
    # A user's file with the bytes of an environment record is a file of its own.
    mine = ws / "data" / "same_as_env.txt"
    mine.write_bytes(Path(str(ln.Run.get(uid=runs[0]).environment.path)).read_bytes())
    eng.begin_run(SessionInfo("s4", ws, "0.0.1"))
    linked = eng.link_input(mine)
    ref = eng.register_artifact(
        mine, key="data/same_as_env.txt", kind="dataset", otype=None, description="d", meta={}
    )
    hidden = ln.Run.get(uid=runs[0]).environment
    emit(envs=envs, n_run_artifacts=ln.Artifact.filter(kind="__lamindb_run__").count(),
         kind=lamin_mod.RUN_ARTIFACT_KIND, scans=scans,
         overview=eng.overview()["artifacts"], listed=[a["key"] for a in eng.list_artifacts()],
         env_files=sorted(p.name for p in (ws / ".lobster" / "environments").iterdir()),
         ws_tree=ws_tree, linked=linked, same=[ref.uid == hidden.uid, ref.existing],
         hidden_key=ln.Artifact.get(hidden.uid).key,
         mine=ln.Artifact.get(ref.uid).kind)
"""


def test_a_run_records_its_environment_once_per_content(tmp_path):
    ws = tmp_path / "ws"
    r = _run(ENVIRONMENT, tmp_path, env_extra={"WS": str(ws)})
    assert r["kind"] == "__lamindb_run__"  # LaminDB's own kind
    one, one_again, two = r["envs"]
    assert one[1:3] == one_again[1:3] == [one[1], "__lamindb_run__"]  # same record
    assert one[1] != two[1] and r["n_run_artifacts"] == 2
    assert [e[4] for e in r["envs"]] == [
        "# one\na==1\n",
        "# one\na==1\n",
        "# two\na==2\n",
    ]
    # The file name is the content hash, inside the workspace's own folder.
    assert r["env_files"] == sorted(f"{e[3]}.txt" for e in (one, two))
    # Nothing else is written: the data files and the two environment files.
    assert r["ws_tree"] == sorted(
        ["data/a.csv", "data/b.csv"]
        + [f".lobster/environments/{e[3]}.txt" for e in (one, two)]
    )
    assert r["scans"] == ["0.0.0", "0.0.1"]  # once per version in this process
    # Hidden from what the user sees: only the data files are listed.
    assert r["overview"] == 3 and sorted(r["listed"]) == [
        "data/a.csv",
        "data/b.csv",
        "data/same_as_env.txt",
    ]
    # A user's file equal to an environment text is not linked to it or merged into it.
    assert r["linked"] is None and r["same"] == [False, False]
    assert "same_as_env" not in r["hidden_key"]  # still the environment file
    assert r["mine"] != "__lamindb_run__"
    assert r["net"] == [] and _tree(tmp_path / "home") == []


def test_a_failed_environment_capture_costs_the_run_nothing(tmp_path):
    code = """
        from pathlib import Path
        import lobster.core.catalog.lamin as lamin_mod
        from lobster.core.catalog.lamin import LaminCatalog
        from lobster.core.catalog.port import SessionInfo
        ws = Path(os.environ["WS"]).resolve()
        ws.mkdir()
        def boom(lobster_version):
            raise RuntimeError("no metadata")
        lamin_mod.environment_text = boom
        eng = LaminCatalog(ws)
        eng.begin_run(SessionInfo("s1", ws, "0.0.0"))
        uid = eng._run.uid
        eng.end_run("completed")
        run = eng.ln.Run.get(uid=uid)
        emit(status=run.status, env=run.environment_id,
             n=eng.ln.Artifact.filter(kind="__lamindb_run__").count())
    """
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert (r["status"], r["env"], r["n"]) == ("completed", None, 0)


def test_a_run_keeps_why_it_did_not_complete_and_an_open_run_has_no_end(tmp_path):
    code = """
        from pathlib import Path
        from lobster.core.catalog.lamin import LaminCatalog
        from lobster.core.catalog.port import SessionInfo
        ws = Path(os.environ["WS"]).resolve()
        ws.mkdir()
        eng = LaminCatalog(ws)
        eng.begin_run(SessionInfo("failed", ws, "0.0.0"))
        eng.end_run("errored", reason="exit code 1")
        eng.begin_run(SessionInfo("done", ws, "0.0.0"))
        eng.end_run("completed")
        eng.begin_run(SessionInfo("left", ws, "0.0.0"))
        eng.begin_run(SessionInfo("next", ws, "0.0.0"))  # nobody ended "left"
        rows = {r["name"]: r for r in eng.list_runs(10)}
        emit(rows={n: [r["status"], r["end_reason"], r["finished_at"] is None]
                   for n, r in rows.items()})
    """
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert r["rows"] == {
        "failed": ["errored", "exit code 1", False],
        "done": ["completed", None, False],
        "left": ["started", None, True],  # open: never completed by a switch
        "next": ["started", None, True],
    }


REPORTS = """
    from pathlib import Path
    import lobster.core.catalog.lamin as lamin_mod
    from lobster.core.catalog.lamin import LaminCatalog, _lamin_hash
    from lobster.core.catalog.port import SessionInfo
    ws = Path(os.environ["WS"]).resolve()
    log = ws / ".lobster" / "sessions" / "s1" / "provenance.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text('{"step": 1}\\n{"step": 2}\\n{"ste')  # a writer is mid-line
    eng = LaminCatalog(ws)
    uids = []

    def run(sid, **end):
        eng.begin_run(SessionInfo(sid, ws, "0.0.0", session_dir=log.parent))
        uids.append(eng._run.uid)
        eng.end_run("completed", **end)

    run("s1", report_path=log)
    log.write_text('{"step": 1}\\n{"step": 2}\\n{"step": 3}\\n')  # the live log grows on
    lamin_mod.REPORT_MAX_BYTES = 10
    run("s2", report_path=log)  # over the cap
    lamin_mod.REPORT_MAX_BYTES = 8 * 1024 * 1024
    run("s3")  # no step log
    ln = eng.ln
    first, over, none = (ln.Run.get(uid=u) for u in uids)
    rep = first.report
    stored = Path(str(rep.path))
    emit(uid=uids[0], kind=rep.kind, description=rep.description, text=stored.read_text(),
         hash_ok=rep.hash == _lamin_hash(stored), stored=str(stored.relative_to(ws)),
         over=over.report_id, none=none.report_id,
         status=[r.status for r in (first, over, none)], live=log.read_text(),
         reports=sorted(p.name for p in (ws / ".lobster" / "reports").iterdir()),
         env_hash=first.environment.hash,
         ws_tree=sorted(str(p.relative_to(ws)) for p in ws.rglob("*") if p.is_file()
                        and not str(p.relative_to(ws)).startswith(".lamindb")),
         overview=eng.overview()["artifacts"], listed=eng.list_artifacts())
"""


def test_a_run_keeps_a_snapshot_of_its_step_log(tmp_path):
    r = _run(REPORTS, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    uid = r["uid"]
    assert r["kind"] == "__lamindb_run__" and r["description"] == f"Report of run {uid}"
    # What the log held when the run ended, up to its last complete line; the live
    # log kept growing.
    assert r["text"] == '{"step": 1}\n{"step": 2}\n' and r["hash_ok"]
    assert r["live"].count("\n") == 3
    assert r["stored"] == f".lobster/reports/{uid}.jsonl"
    # Over the size cap, or no log given: no report, and the run still completed.
    assert r["over"] is None and r["none"] is None
    assert r["status"] == ["completed"] * 3
    assert r["reports"] == [f"{uid}.jsonl"]
    # The only files the catalog adds to the workspace: the two records' files.
    assert r["ws_tree"] == [
        f".lobster/environments/{r['env_hash']}.txt",
        f".lobster/reports/{uid}.jsonl",
        ".lobster/sessions/s1/provenance.jsonl",
    ]
    assert r["overview"] == 0 and r["listed"] == []  # not a file the user saved
    assert r["net"] == [] and _tree(tmp_path / "home") == []


# Made of pieces so that no secret scanner mistakes a test for a leak.
SECRET_LOGS = {
    "aws": "key " + "AKIA" + "ABCDEFGHIJKLMNOP",
    "anthropic": "sk-" + "ant-api03-abcdefghij",
    "openai": "sk-" + "abcdefghijklmnopqrstuvwx",
    "github_classic": "ghp" + "_abcdefghijklmnopqrstuvwxyz0123456789",
    "github_fine": "github" + "_pat_11ABCDEFG0abcdefghij",
    "private_key": "-----BEGIN RSA PRIVATE" + " KEY-----\\nMIIEabc\\n",
    "bearer": "curl -H 'Authorization: Bearer abcdefghijklmnopqrstu.vwxyz'",
    "assignment": "export password=hunter2hunter2hunter2",
    "assignment_quoted": 'api_key = "abcdefghijklmnop"',
    "lamin": "LAMIN_API_KEY: abcdefghijkl",
    # More key forms
    "openai_project": "sk-" + "proj-abcdefghijklmnopqrstuvwxyz0123",
    "openrouter": "sk-" + "or-v1-abcdefghijklmnopqrstuvwx",
    "aws_sts": "ASIA" + "ABCDEFGHIJKLMNOP",
    "slack": "xox" + "b-1234567890-abcdefghij",
    "aws_secret_name": "AWS_SECRET_ACCESS_KEY=abcdefghijklmnopqrstuvwx",
    "secret_key": 'secret_key: "abcdefghijklmnop"',
    "header_name": "x-api-key: abcdefghijklmnop",
    "access_token": '"access_token": "abcdefghijklmnop"',
    "plural_tokens": "tokens=abcdefghijklmnop",
    "upper_case": "API_KEY=abcdefghijklmnop and PASSWORD=hunter2hunter2hunter2",
}
CLEAN_LOGS = {
    "hash": "sha256 " + "0123456789abcdef" * 4,  # a 64-hex hash alone is no secret
    "token_counts": 'max_tokens: 4096 and "tokens": 4096 and token_count: 123',
    "password_column": "password_column: user_password_col",
    # One benign string per newer pattern: the name or prefix without a token.
    "asia_in_prose": "the ASIA region and ASIAN markets; ASIA-PACIFIC",
    "xox_in_words": "xox and xoxo, a boxoxen, xoxb-123 is too short",
    "short_sk_tokens": "sk-short, sk-proj-abc and a task-list item",
    # ``sk-`` inside a word is no key: slugs of papers, tasks and disks.
    "sk_inside_words": "risk-adjusted-return-series-v2, task-based-workflow-model-v3, "
    "disk-backed-store-of-large-matrices, task-ant-colony",
    "secret_key_in_prose": "pass secret_key as an argument; secret_key: see docs",
    "header_name_only": "send the x-api-key header; x-api-key: required",
    "name_without_value": "set AWS_SECRET_ACCESS_KEY in the environment; tokens: none",
    # Short or spaced passwords stay out: a deliberate limit of the scan.
    "short_values": '"token": "short" and password=abc123 and secret: yes',
    "plain": "normalised 3 samples with pyDESeq2 at /data/proc.h5ad",
}


# What a warning must never carry: the part of a secret that a pattern matches (some match
# only the prefix) and the part of the value after it. Whole lines are not enough: a
# message that leaks only the matched part would not contain the line around it.
SECRET_PARTS = (
    "ABCDEFGHIJKLMNOP",  # an AKIA or an ASIA key
    "sk-" + "ant-",
    "ghp" + "_",
    "github" + "_pat_",
    "11ABCDEFG0",
    "xox" + "b-1234567890",
    "-----BEGIN RSA PRIVATE" + " KEY-----",
    "MIIEabc",
    "hunter2hunter2",
    "abcdefghijkl",  # the value of most of the others
)


def test_every_secret_log_has_a_part_a_message_must_not_carry():
    assert all(any(p in text for p in SECRET_PARTS) for text in SECRET_LOGS.values())
    assert all(any(p in text for text in SECRET_LOGS.values()) for p in SECRET_PARTS)


def test_a_step_log_that_looks_like_it_holds_a_secret_is_not_recorded(tmp_path):
    """Decision 11 Nov 2026: secrets scan on, fail closed; the log is never altered."""
    code = """
        import logging
        from pathlib import Path
        from lobster.core.catalog.lamin import LaminCatalog
        from lobster.core.catalog.port import SessionInfo
        ws = Path(os.environ["WS"]).resolve()
        folder = ws / ".lobster" / "sessions" / "s"
        folder.mkdir(parents=True)
        messages = []

        class Keep(logging.Handler):
            def emit(self, record):
                messages.append(record.getMessage())

        logging.getLogger("lobster.core.catalog.lamin").addHandler(Keep())
        eng = LaminCatalog(ws)
        out = {}
        for name, text in json.loads(os.environ["LOGS"]).items():
            log = folder / f"{name}.jsonl"
            log.write_text(json.dumps({"tool": "t", "params": text}) + "\\n")
            before = log.read_bytes()
            eng.begin_run(SessionInfo(name, ws, "0.0.0"))
            uid = eng._run.uid
            eng.end_run("completed", report_path=log)
            run = eng.ln.Run.get(uid=uid)
            out[name] = [run.status, run.report_id is not None, log.read_bytes() == before, uid]
        reports = sorted(p.name for p in (ws / ".lobster" / "reports").glob("*"))
        emit(out=out, reports=reports, messages=messages)
    """
    logs = {**{k: v for k, v in SECRET_LOGS.items()}, **CLEAN_LOGS}
    r = _run(
        code,
        tmp_path,
        env_extra={"WS": str(tmp_path / "ws"), "LOGS": json.dumps(logs)},
    )
    for name in SECRET_LOGS:
        assert r["out"][name][:3] == ["completed", False, True], name
    for name in CLEAN_LOGS:
        assert r["out"][name][:3] == ["completed", True, True], name
    # Only the clean logs left a snapshot, and no snapshot was written for the others.
    assert r["reports"] == sorted(f"{r['out'][n][3]}.jsonl" for n in CLEAN_LOGS)
    # Said once, naming the kind of pattern and never the text that matched.
    said = [m for m in r["messages"] if "looks like it contains a secret" in m]
    assert len(said) == 1 and said[0].startswith("Catalog: Report not recorded")
    assert not any(part in m for m in r["messages"] for part in SECRET_PARTS)
    assert r["net"] == []


def test_a_notebook_that_looks_like_it_holds_a_secret_is_not_registered(tmp_path):
    """Decision 11 Nov 2026: the notebook source gets the step log's scan, fail closed.

    The notebook file is never altered; the run says which notebooks were refused.
    """
    code = """
        import logging
        from pathlib import Path
        from unittest.mock import Mock
        from lobster.core.catalog import factory
        from lobster.core.runtime.data_manager import DataManagerV2
        ws = Path(os.environ["WS"]).resolve()
        messages = []

        class Keep(logging.Handler):
            def emit(self, record):
                messages.append(record.getMessage())

        logging.getLogger("lobster.core.catalog.lamin").addHandler(Keep())
        dm = DataManagerV2(workspace_path=ws)
        catalog = factory.begin_session_run(dm, ws, "session_a")
        dm.log_tool_usage("qc", {}, description="d")
        folder = ws / "notebooks"
        folder.mkdir()
        unchanged = {}
        for name, text in json.loads(os.environ["TEXTS"]).items():
            nb = folder / f"{name}.ipynb"
            nb.write_text(json.dumps({"cells": [{"source": [text]}], "nbformat": 4}))
            before = nb.read_bytes()
            dm._notebook_exporter = Mock(export=Mock(return_value=nb))
            dm.export_notebook("session", description="exported session")
            unchanged[name] = nb.read_bytes() == before
        catalog.close("completed")
        emit(unchanged=unchanged, messages=messages)
    """
    readback = """
        from pathlib import Path
        ws = Path(os.environ["WS"])
        os.environ["LAMIN_SETTINGS_DIR"] = str(ws / ".lamindb" / "settings")
        import lamindb_setup as lns
        lns.connect(json.loads((ws / ".lamindb/lobster.json").read_text())["slug"],
                    _write_settings=False)
        import lamindb as ln
        run = ln.Run.get(name="session_a")
        emit(keys=sorted(t.key for t in ln.Transform.filter(kind="notebook")),
             sources=[t.source_code for t in ln.Transform.filter(kind="notebook")],
             refused=((run.extra_data or {}).get("lobster") or {}).get(
                 "unregistered_secret_scan"),
             status=run.status)
    """
    clean = {
        **CLEAN_LOGS,
        "token_in_prose": "A token is one unit of text; the model reads tokens: "
        "words and pieces of words.",
    }
    texts = {**SECRET_LOGS, **clean}
    ws = tmp_path / "ws"
    ws.mkdir()
    r = _run(
        code,
        tmp_path,
        env_extra={"WS": str(ws), "TEXTS": json.dumps(texts)},
    )
    back = _run(readback, tmp_path, env_extra={"WS": str(ws)})
    assert all(r["unchanged"].values()) and set(r["unchanged"]) == set(texts)
    assert back["status"] == "completed"
    assert back["keys"] == sorted(f"notebooks/{n}.ipynb" for n in clean)
    assert sorted(back["refused"]) == sorted(
        f"notebooks/{n}.ipynb" for n in SECRET_LOGS
    )
    # Said once, naming the kind of pattern and never the text that matched.
    said = [m for m in r["messages"] if "looks like it contains a secret" in m]
    assert len(said) == 1 and said[0].startswith("Catalog: Notebook not recorded")
    assert not any(part in m for m in r["messages"] for part in SECRET_PARTS)
    assert not any(
        text in (src or "") for src in back["sources"] for text in SECRET_LOGS.values()
    )
    assert r["net"] == back["net"] == []


def test_a_notebook_refused_twice_is_listed_once_on_the_run(tmp_path):
    """Exporting the same refused notebook again does not repeat it in the run's list."""
    code = """
        from pathlib import Path
        from unittest.mock import Mock
        from lobster.core.catalog import factory
        from lobster.core.runtime.data_manager import DataManagerV2
        ws = Path(os.environ["WS"]).resolve()
        dm = DataManagerV2(workspace_path=ws)
        catalog = factory.begin_session_run(dm, ws, "session_a")
        dm.log_tool_usage("qc", {}, description="d")
        folder = ws / "notebooks"
        folder.mkdir()
        for name in ["a", "a", "b", "a"]:
            nb = folder / f"{name}.ipynb"
            nb.write_text(json.dumps({"cells": [{"source": [os.environ["SECRET"]]}],
                                      "nbformat": 4}))
            dm._notebook_exporter = Mock(export=Mock(return_value=nb))
            dm.export_notebook("session", description="exported session")
        catalog.close("completed")
        emit()
    """
    readback = """
        from pathlib import Path
        ws = Path(os.environ["WS"])
        os.environ["LAMIN_SETTINGS_DIR"] = str(ws / ".lamindb" / "settings")
        import lamindb_setup as lns
        lns.connect(json.loads((ws / ".lamindb/lobster.json").read_text())["slug"],
                    _write_settings=False)
        import lamindb as ln
        run = ln.Run.get(name="session_a")
        emit(refused=((run.extra_data or {}).get("lobster") or {}).get(
            "unregistered_secret_scan"))
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    env = {"WS": str(ws), "SECRET": SECRET_LOGS["aws"]}
    _run(code, tmp_path, env_extra=env)
    back = _run(readback, tmp_path, env_extra=env)
    assert back["refused"] == ["notebooks/a.ipynb", "notebooks/b.ipynb"]


def test_a_report_that_cannot_be_saved_leaves_no_copy_behind(tmp_path):
    code = """
        from pathlib import Path
        from lobster.core.catalog.lamin import LaminCatalog
        from lobster.core.catalog.port import SessionInfo
        ws = Path(os.environ["WS"]).resolve()
        log = ws / ".lobster" / "sessions" / "s1" / "provenance.jsonl"
        log.parent.mkdir(parents=True)
        log.write_text('{"step": 1}\\n')
        eng = LaminCatalog(ws)
        eng.begin_run(SessionInfo("s1", ws, "0.0.0"))
        uid = eng._run.uid
        save = eng.ln.Artifact.save

        def failing(self, *args, **kwargs):
            if (self.description or "").startswith("Report of run"):
                raise OSError("no space left")
            return save(self, *args, **kwargs)

        eng.ln.Artifact.save = failing
        eng.end_run("completed", report_path=log)
        eng.ln.Artifact.save = save
        run = eng.ln.Run.get(uid=uid)
        reports = ws / ".lobster" / "reports"
        emit(status=run.status, report=run.report_id, env=run.environment_id is not None,
             left=sorted(p.name for p in reports.iterdir()) if reports.exists() else [])
    """
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert (r["status"], r["report"], r["env"], r["left"]) == (
        "completed",
        None,
        True,
        [],
    )


def test_the_secret_scan_is_linear_and_still_finds_private_keys():
    """A pattern that backtracks would freeze every thread (the ``re`` module holds
    the GIL) and outlast the exit budget: 4000 BEGIN markers on one 1 MiB line took
    2.8 s with ``-----BEGIN .*PRIVATE KEY-----``. CPU time, not wall time: a loaded
    machine must not fail the test, a backtracking pattern must."""
    import time

    from lobster.core.catalog.lamin import REPORT_SECRET_PATTERNS

    mib = 1024 * 1024

    def repeat(unit):
        return unit * (mib // len(unit))

    hostile = {
        "begin markers": "-----BEGIN " * 4000 + "x" * (mib - 44000),
        "bearer stubs": repeat("Bearer " + "a" * 19 + " "),
        "key stubs": repeat("token=" + "a" * 11 + " "),
        "sk stubs": repeat("sk-" + "a" * 19),
        "sk chain": repeat("sk-"),
        "sts stubs": repeat("ASIA" + "A" * 15),
        "slack stubs": repeat("xoxb-" + "a" * 9 + " "),
        "keyword chain": repeat("secret_access_key"),
        "keyword stubs": repeat("x-api-key " + "=" + " " * 3),
        "separator runs": "secret" + " " * (mib - 6),
    }
    for name, text in hostile.items():
        for label, pattern in REPORT_SECRET_PATTERNS:
            started = time.process_time()
            pattern.search(text)
            assert time.process_time() - started < 0.5, (name, label)
    private = next(p for label, p in REPORT_SECRET_PATTERNS if label == "private key")
    for kind in ("RSA ", "EC ", "OPENSSH ", "ENCRYPTED ", "DSA ", ""):
        assert private.search(f"x -----BEGIN {kind}PRIVATE" + " KEY-----\\nabc"), kind


def test_the_run_is_closed_before_its_environment_is_captured(tmp_path):
    """A slow or failed capture at exit must not leave the run open."""
    code = """
        from pathlib import Path
        import lobster.core.catalog.lamin as lamin_mod
        from lobster.core.catalog.lamin import LaminCatalog
        from lobster.core.catalog.port import SessionInfo
        ws = Path(os.environ["WS"]).resolve()
        ws.mkdir()
        eng = LaminCatalog(ws)
        eng.begin_run(SessionInfo("s1", ws, "0.0.0"))
        uid = eng._run.uid
        seen = {}
        original = LaminCatalog._environment
        def spy(self, lobster_version):
            run = self.ln.Run.get(uid=uid)
            seen["status"], seen["finished"] = run.status, run.finished_at is not None
            return original(self, lobster_version)
        LaminCatalog._environment = spy
        eng.end_run("errored")
        run = eng.ln.Run.get(uid=uid)
        emit(seen=seen, status=run.status, env=run.environment_id is not None)
    """
    r = _run(code, tmp_path, env_extra={"WS": str(tmp_path / "ws")})
    assert r["seen"] == {"status": "errored", "finished": True}
    assert r["status"] == "errored" and r["env"]


def test_two_workspaces_on_one_instance_share_one_environment_file(tmp_path):
    """A named instance: no copy without a record, whichever workspace saw the environment first."""
    inst = tmp_path / "inst"
    created = _run(
        f"""
        import lamindb_setup
        lamindb_setup.init(storage={str(inst)!r}, name="shared2")
        emit(slug=lamindb_setup.settings.instance.slug)
        """,
        tmp_path,
    )
    code = """
        from pathlib import Path
        import lobster.core.catalog.lamin as lamin_mod
        from lobster.core.catalog.lamin import LaminCatalog
        from lobster.core.catalog.port import SessionInfo
        inst = Path(os.environ["INST"]).resolve()
        # With the switch off, package names do not leave the workspace.
        lamin_mod.INSTANCE_ENVIRONMENT = False
        ws = Path(os.environ["BASE"]) / "ws_off"
        ws.mkdir()
        eng = LaminCatalog(ws, instance="anonymous/shared2")
        eng.begin_run(SessionInfo("s_off", ws, "0.0.0"))
        uid = eng._run.uid
        eng.end_run("completed")
        switched_off = [eng.ln.Run.get(uid=uid).environment_id,
                        sorted(str(p) for p in (inst / "lobster").rglob("*.txt"))]
        lamin_mod.INSTANCE_ENVIRONMENT = True  # the default, restored by name
        runs = []
        for name in ("one", "two", "one"):
            ws = Path(os.environ["BASE"]) / f"ws_{name}"
            ws.mkdir(exist_ok=True)
            eng = LaminCatalog(ws, instance="anonymous/shared2")
            eng.begin_run(SessionInfo(f"s_{len(runs)}", ws, "0.0.0"))
            uid = eng._run.uid
            eng.end_run("completed")
            runs.append(eng.ln.Run.get(uid=uid).environment_id)
        files = sorted(str(p.relative_to(inst)) for p in (inst / "lobster").rglob("*.txt"))
        records = [a.hash for a in eng.ln.Artifact.filter(kind="__lamindb_run__")]
        # A user's file with the bytes of the environment record is a file of its own.
        hidden = eng.ln.Artifact.filter(kind="__lamindb_run__").first()
        mine = ws / "same_as_env.txt"
        mine.write_bytes(Path(str(hidden.path)).read_bytes())
        eng.begin_run(SessionInfo("s_mine", ws, "0.0.0"))
        linked = eng.link_input(mine)
        ref = eng.register_artifact(
            mine, key="same_as_env.txt", kind="dataset", otype=None, description="d", meta={}
        )
        emit(runs=runs, files=files, records=records, switched_off=switched_off,
             linked=linked, same=[ref.uid == hidden.uid, ref.existing],
             mine=eng.ln.Artifact.get(ref.uid).kind,
             hidden_key=eng.ln.Artifact.get(hidden.uid).key)
    """
    r = _run(code, tmp_path, env_extra={"BASE": str(tmp_path), "INST": str(inst)})
    assert r["switched_off"] == [None, []]
    assert len(set(r["runs"])) == 1 and r["runs"][0] is not None  # one record
    assert [Path(f).stem for f in r["files"]] == r["records"]  # and one file
    assert r["linked"] is None and r["same"] == [False, False]
    assert "same_as_env" not in r["hidden_key"]  # still the environment file
    assert r["mine"] != "__lamindb_run__"
    assert r["net"] == []


def test_a_run_is_created_in_one_place():
    """Decision (Nov 2026): the single place a per-tool Run policy would plug in."""
    import ast
    import inspect

    from lobster.core.catalog import lamin

    def creates_a_run(node):
        if not isinstance(node, ast.Call):
            return False
        f = node.func
        if isinstance(f, ast.Name) and f.id == "Run":  # Run(...)
            return True
        if isinstance(f, ast.Attribute) and f.attr == "Run":  # ln.Run(...)
            return True
        # Run.objects.create(...) / ln.Run.objects.get_or_create(...)
        return (
            isinstance(f, ast.Attribute)
            and f.attr in ("create", "get_or_create", "bulk_create")
            and isinstance(f.value, ast.Attribute)
            and f.value.attr == "objects"
            and "Run" in ast.unparse(f.value.value)
        )

    tree = ast.parse(inspect.getsource(lamin))
    owners = [
        fn.name
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        for node in ast.walk(fn)
        if creates_a_run(node)
    ]
    # a replay is the other creator: a Run of a notebook, never a session Run
    assert owners == ["_new_run", "begin_replay"]
    assert "_new_run(" in inspect.getsource(lamin.LaminCatalog.begin_run)


def test_a_write_that_fails_leaves_no_partial_file(tmp_path):
    from lobster.core.catalog.lamin import _write_whole

    with pytest.raises(TypeError):
        _write_whole(tmp_path / "sub" / "x.txt", "not bytes")  # type: ignore[arg-type]
    assert list((tmp_path / "sub").iterdir()) == []


def test_a_record_outside_the_storage_is_never_written_to(tmp_path):
    """The environment record is found by hash; its file must be inside the storage."""
    code = """
        from pathlib import Path
        import lobster.core.catalog.lamin as lamin_mod
        from lobster.core.catalog.lamin import LaminCatalog
        from lobster.core.catalog.port import SessionInfo
        ws = Path(os.environ["WS"]).resolve()
        ws.mkdir()
        outside = Path(os.environ["OUT"]).resolve() / "stray.txt"
        outside.parent.mkdir()

        class Record:
            path = outside

        class Found:
            def first(self):
                return Record()

        eng = LaminCatalog(ws)
        eng.begin_run(SessionInfo("s1", ws, "0.0.0"))
        uid = eng._run.uid
        real = eng.ln.Artifact.filter

        def filter_(*args, **kwargs):
            if kwargs.get("kind") == lamin_mod.RUN_ARTIFACT_KIND and "hash" in kwargs:
                return Found()
            return real(*args, **kwargs)

        eng.ln.Artifact.filter = filter_
        eng.end_run("completed")
        eng.ln.Artifact.filter = real
        run = eng.ln.Run.get(uid=uid)
        emit(status=run.status, env=run.environment_id,
             outside=sorted(p.name for p in outside.parent.iterdir()))
    """
    r = _run(
        code,
        tmp_path,
        env_extra={"WS": str(tmp_path / "ws"), "OUT": str(tmp_path / "out")},
    )
    assert (r["status"], r["env"], r["outside"]) == ("completed", None, [])


def test_an_instance_not_set_up_here_says_so_offline(tmp_path):
    code = """
        from pathlib import Path
        from lobster.core.catalog import factory
        ws = Path(os.environ["WS"]).resolve()
        ws.mkdir()
        st = factory.get_catalog(ws).status()
        emit(degraded=st.degraded, reason=st.reason)
    """
    r = _run(
        code,
        tmp_path,
        env_extra={
            "WS": str(tmp_path / "ws"),
            "LOBSTER_LAMIN_INSTANCE": "anonymous/elsewhere",
        },
    )
    assert r["degraded"]
    assert "anonymous/elsewhere is not set up on this machine" in r["reason"]
    assert r["net"] == []


def test_a_named_instance_with_remote_storage_is_refused(tmp_path):
    """Only local instance storage is supported: a remote one is refused, offline."""
    inst = tmp_path / "inst"
    _run(
        f"""
        import lamindb_setup
        lamindb_setup.init(storage={str(inst)!r}, name="remote")
        emit()
        """,
        tmp_path,
    )
    code = """
        from pathlib import Path
        import lamindb_setup
        from lamindb_setup.core._settings_storage import StorageSettings
        from lobster.core.catalog import factory
        connect = lamindb_setup.connect

        def connect_remote(*args, **kwargs):  # connected: storage reads as s3://
            connect(*args, **kwargs)
            StorageSettings.type = property(lambda self: "s3")
            StorageSettings.root = property(lambda self: "s3://lobster-test/inst")

        lamindb_setup.connect = connect_remote
        ws = Path(os.environ["WS"]).resolve()
        (ws / "data").mkdir(parents=True)
        f = ws / "data" / "a.csv"
        f.write_text("x\\n1\\n")
        cat = factory.get_catalog(ws)
        cat.register_artifact(f, key="data/a.csv", kind="dataset", otype=None,
                              description="d", meta={})
        st = cat.status()
        inst = Path(os.environ["INST"])
        emit(degraded=st.degraded, reason=st.reason,
             stray=sorted(str(p) for top in (Path.cwd(), inst, ws)
                          for p in top.rglob("*") if p.name in ("s3:", "lobster")))
    """
    r = _run(
        code,
        tmp_path,
        env_extra={
            "WS": str(tmp_path / "ws"),
            "INST": str(inst),
            "LOBSTER_LAMIN_INSTANCE": "anonymous/remote",
        },
    )
    assert r["degraded"]
    assert "records only into local storage; the LaminDB instance " in r["reason"]
    assert "anonymous/remote is remote (s3://lobster-test/inst)" in r["reason"]
    assert r["stray"] == [] and r["net"] == []


def test_an_instance_whose_settings_name_remote_storage_is_refused_offline(tmp_path):
    """Read from the user's settings file before connecting: no network attempt."""
    inst = tmp_path / "inst"
    created = _run(
        f"""
        from pathlib import Path
        import lamindb_setup
        lamindb_setup.init(storage={str(inst)!r}, name="cloudy")
        settings = Path(os.environ["HOME"]) / ".lamin" / "instance--anonymous--cloudy.env"
        text = settings.read_text().replace(
            "lamindb_instance_storage_root={inst.resolve()}",
            "lamindb_instance_storage_root=s3://lobster-test/inst",
        )
        settings.write_text(text)
        emit(edited="s3://lobster-test/inst" in settings.read_text())
        """,
        tmp_path,
    )
    assert created["edited"]
    code = """
        from pathlib import Path
        from lobster.core.catalog import factory
        ws = Path(os.environ["WS"]).resolve()
        (ws / "data").mkdir(parents=True)
        f = ws / "data" / "a.csv"
        f.write_text("x\\n1\\n")
        cat = factory.get_catalog(ws)
        ref = cat.register_artifact(f, key="data/a.csv", kind="dataset", otype=None,
                                    description="d", meta={})
        st = cat.status()
        inst = Path(os.environ["INST"])
        emit(degraded=st.degraded, reason=st.reason, ref=ref is None,
             stray=sorted(str(p) for top in (Path.cwd(), inst, ws)
                          for p in top.rglob("*") if p.name in ("s3:", "lobster")))
    """
    r = _run(
        code,
        tmp_path,
        env_extra={
            "WS": str(tmp_path / "ws"),
            "INST": str(inst),
            "LOBSTER_LAMIN_INSTANCE": "anonymous/cloudy",
        },
    )
    assert r["degraded"] and r["ref"]
    assert "records only into local storage; the LaminDB instance " in r["reason"]
    assert "anonymous/cloudy is remote (s3://lobster-test/inst)" in r["reason"]
    assert r["stray"] == [] and r["net"] == []


@pytest.mark.parametrize(
    "root,remote",
    [
        ("s3://bucket/x", True),
        ("gs://bucket/x", True),
        ("create-s3", True),
        ("create-s3--eu-central-1", True),
        ("/Users/me/inst", False),
        ("file:///Users/me/inst", False),
        ("C:\\lamin\\inst", False),
    ],
)
def test_remote_storage_roots_are_recognised_before_connect(root, remote):
    from lobster.core.catalog.lamin import _is_remote_root

    assert _is_remote_root(root) is remote


def test_ls_and_status_follow_lamin_use_and_clear(tmp_path):
    """After ``lamin use`` the listing and the counts are the instance's."""
    store = tmp_path / "store"
    store.mkdir()
    created = _run(
        """
        import lamindb_setup as lns
        lns.init(storage=os.environ["STORE"], name="shared")
        emit(slug=lns.settings.instance.slug)
        """,
        tmp_path,
        env_extra={"STORE": str(store)},
    )
    ws = tmp_path / "ws"
    ws.mkdir()
    env = {"WS": str(ws), "SLUG": created["slug"], "STORE": str(store)}
    record = """
        from pathlib import Path
        from lobster.core.catalog import factory
        from lobster.core.runtime.data_manager import DataManagerV2
        ws = Path(os.environ["WS"])
        dm = DataManagerV2(workspace_path=ws)
        catalog = factory.begin_session_run(dm, ws, os.environ["SESSION"])
        catalog.close("completed")
        emit(type=type(catalog).__name__)
    """
    _run(  # a session in the workspace's own catalog
        "from typer.testing import CliRunner\n"
        "from lobster.cli_internal.commands.light.lamin_commands import lamin_app\n"
        "assert CliRunner().invoke(lamin_app, ['init', '-w', os.environ['WS']])"
        ".exit_code == 0\n" + textwrap.dedent(record),
        tmp_path,
        env_extra={**env, "SESSION": "session_workspace"},
        drop_enable=True,
    )
    _run(  # a session in the instance, after `lamin use`
        "from typer.testing import CliRunner\n"
        "from lobster.cli_internal.commands.light.lamin_commands import lamin_app\n"
        "assert CliRunner().invoke(lamin_app, ['use', os.environ['SLUG'], '-w', "
        "os.environ['WS']]).exit_code == 0\n" + textwrap.dedent(record),
        tmp_path,
        env_extra={**env, "SESSION": "session_instance"},
        drop_enable=True,
    )
    look = """
        from typer.testing import CliRunner
        from lobster.cli_internal.commands.light.lamin_commands import lamin_app
        cli = CliRunner()
        def run(*args):
            result = cli.invoke(lamin_app, [*args, "-w", os.environ["WS"]])
            assert result.exit_code == 0, result.output
            return result.stdout
        emit(runs=[r["name"] for r in json.loads(run("ls", "--runs", "--json"))],
             status=json.loads(run("status", "--json")), text=run("status"))
    """
    pruned = _run(
        "from typer.testing import CliRunner\n"
        "from lobster.cli_internal.commands.light.lamin_commands import lamin_app\n"
        "r = CliRunner().invoke(lamin_app, ['prune', '--dry-run', '--json', '-w', "
        "os.environ['WS']])\nemit(code=r.exit_code, out=r.stdout)",
        tmp_path,
        env_extra=env,
        drop_enable=True,
    )
    assert pruned["code"] == 0 and '"files": 0' in pruned["out"]
    used = _run(look, tmp_path, env_extra=env, drop_enable=True)
    assert used["runs"] == ["session_instance"]
    assert used["status"]["contents"]["last_run"]["name"] == "session_instance"
    assert "Workspace catalog (not in use)" not in used["text"]
    assert "session_workspace" not in used["text"]
    _run(
        "from typer.testing import CliRunner\n"
        "from lobster.cli_internal.commands.light.lamin_commands import lamin_app\n"
        "assert CliRunner().invoke(lamin_app, ['use', '--clear', '-w', "
        "os.environ['WS']]).exit_code == 0\nemit(done=True)",
        tmp_path,
        env_extra=env,
        drop_enable=True,
    )
    cleared = _run(look, tmp_path, env_extra=env, drop_enable=True)
    assert cleared["runs"] == ["session_workspace"]


def test_an_instance_not_set_up_here_is_refused_by_ls_status_and_prune(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    code = """
        from typer.testing import CliRunner
        from lobster.cli_internal.commands.light.lamin_commands import lamin_app
        cli = CliRunner()
        out = {}
        for name in ("ls", "status", "prune"):
            result = cli.invoke(lamin_app, [name, "-w", os.environ["WS"]])
            out[name] = [result.exit_code, result.output + (result.stderr or "")]
        emit(out=out)
    """
    r = _run(
        code,
        tmp_path,
        env_extra={"WS": str(ws), "LOBSTER_LAMIN_INSTANCE": "nobody/ghost"},
        drop_enable=True,
    )
    for name, (exit_code, text) in r["out"].items():
        assert exit_code == 1, (name, text)
        assert "not set up on this machine" in text, (name, text)
