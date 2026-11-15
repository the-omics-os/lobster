"""One catalog for a project: its own workspace and every run workspace share it.

The recording itself (LaminDB) is pinned in ``test_lamin_project_runs.py``; these
tests use a fake engine and need no extra.
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from lobster.core.catalog import factory
from lobster.core.catalog.factory import get_catalog, resolve_catalog_config
from lobster.core.catalog.layout import catalog_layout
from lobster.core.catalog.null import NullCatalog
from lobster.core.catalog.thread import ThreadedCatalog
from lobster.core.projects import init_project

pytestmark = pytest.mark.unit

ON = {"LOBSTER_LAMIN": "1"}


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    init_project(root, "study")
    return root.resolve()


def run_workspace(project, name):
    path = project / "runs" / name / ".lobster_workspace"
    path.mkdir(parents=True)
    return path


def switch_on(project):
    (project / ".lobster_workspace" / ".env").write_text("LOBSTER_LAMIN=1\n")


def made_before(project):
    """The catalog an earlier version made in the project's own workspace."""
    folder = project / ".lobster_workspace" / ".lamindb"
    folder.mkdir()
    (folder / "lobster.json").write_text('{"slug": "anonymous/old"}')
    return folder


# -- where a catalog lives ------------------------------------------------------


def test_a_plain_workspace_keeps_its_catalog_inside_itself(tmp_path):
    ws = tmp_path / "plain" / ".lobster_workspace"
    ws.mkdir(parents=True)
    layout = catalog_layout(ws)
    assert (layout.home, layout.root, layout.project) == (ws, ws, None)
    assert layout.catalog_dir == ws / ".lamindb" and layout.runs_recorded is None


def test_the_project_and_its_runs_share_one_catalog_in_the_project_folder(project):
    own = project / ".lobster_workspace"
    first, second = run_workspace(project, "a"), run_workspace(project, "b")
    layouts = [catalog_layout(p) for p in (own, first, second)]
    assert {(x.home, x.root, x.project) for x in layouts} == {(own, project, project)}
    assert layouts[1].catalog_dir == project / ".lamindb"
    assert layouts[1].workspace == first and layouts[1].runs_recorded is True


def test_a_workspace_outside_the_runs_folder_is_not_the_projects(project):
    stray = project / "elsewhere" / ".lobster_workspace"
    stray.mkdir(parents=True)
    assert catalog_layout(stray).project is None


def test_an_old_project_catalog_keeps_serving_the_project_but_not_its_runs(project):
    old = made_before(project)
    own, run = project / ".lobster_workspace", run_workspace(project, "a")
    for ws in (own, run):
        layout = catalog_layout(ws)
        assert layout.legacy and layout.root == own and layout.runs_recorded is False
        assert layout.catalog_dir == old
    fix = catalog_layout(run).legacy_fix
    assert fix.startswith("mv ") and str(old) in fix


def test_a_run_that_made_its_own_catalog_keeps_it(project):
    run = run_workspace(project, "a")
    (run / ".lamindb").mkdir()
    (run / ".lamindb" / "lobster.json").write_text("{}")
    layout = catalog_layout(run)
    assert (layout.root, layout.project) == (run, None)


def test_a_named_instance_keys_files_from_the_project_even_over_an_old_catalog(project):
    made_before(project)
    run = run_workspace(project, "a")
    assert catalog_layout(run, instance=True).root == project


# -- the setting reaches the runs -----------------------------------------------


def test_a_run_workspace_inherits_the_projects_catalog_setting(project):
    switch_on(project)
    config = resolve_catalog_config(
        run_workspace(project, "a"), environ={}, global_config_path=Path("none")
    )
    assert (config.enabled, config.source) == (True, "project")


def test_a_run_can_still_switch_the_catalog_off_for_itself(project):
    switch_on(project)
    run = run_workspace(project, "a")
    (run / ".env").write_text("LOBSTER_LAMIN=0\n")
    config = resolve_catalog_config(run, environ={}, global_config_path=Path("none"))
    assert (config.enabled, config.source) == (False, "workspace")


def test_the_shell_still_beats_the_project(project):
    switch_on(project)
    config = resolve_catalog_config(
        run_workspace(project, "a"),
        environ={"LOBSTER_LAMIN": "0"},
        global_config_path=Path("none"),
    )
    assert (config.enabled, config.source) == (False, "env")


def test_the_projects_instance_setting_reaches_its_runs(project):
    (project / ".lobster_workspace" / ".env").write_text(
        "LOBSTER_LAMIN=1\nLOBSTER_LAMIN_INSTANCE=lab/shared\n"
    )
    config = resolve_catalog_config(
        run_workspace(project, "a"), environ={}, global_config_path=Path("none")
    )
    assert (config.instance, config.instance_source) == ("lab/shared", "project")


def test_a_plain_workspace_does_not_read_a_project_it_is_not_in(tmp_path):
    ws = tmp_path / "plain" / ".lobster_workspace"
    ws.mkdir(parents=True)
    config = resolve_catalog_config(ws, environ={}, global_config_path=Path("none"))
    assert config.enabled is False


# -- one process, many runs -----------------------------------------------------


@pytest.fixture
def engines(monkeypatch, recording_catalog_cls):
    built = []

    def engine_class(workspace, instance=None):
        engine = recording_catalog_cls(workspace, instance)
        built.append(engine)
        return engine

    monkeypatch.setattr(factory, "_load_engine", lambda: engine_class)
    # Another test may have loaded LaminDB in this process, with other settings.
    monkeypatch.setattr(
        "lobster.core.catalog.lamin.workspace_settings_conflict", lambda workspace: None
    )
    return built


def test_every_run_of_a_project_gets_the_same_catalog(project, engines):
    switch_on(project)
    first = get_catalog(run_workspace(project, "a"), environ={})
    second = get_catalog(run_workspace(project, "b"), environ={})
    own = get_catalog(project / ".lobster_workspace", environ={})
    assert isinstance(first, ThreadedCatalog)
    assert second is first and own is first
    first.status()
    assert len(engines) == 1


def test_the_catalog_keys_files_from_the_project(project, engines):
    switch_on(project)
    catalog = get_catalog(run_workspace(project, "a"), environ={})
    assert catalog.key_root == project


def test_a_plain_workspace_keys_files_from_itself_as_before(tmp_path, engines):
    ws = tmp_path / "ws"
    ws.mkdir()
    catalog = get_catalog(ws, environ=ON)
    assert isinstance(catalog, ThreadedCatalog) and catalog.key_root == ws.resolve()


def test_a_run_does_not_use_the_catalog_of_another_project_in_the_process(
    project, tmp_path, engines
):
    switch_on(project)
    other = tmp_path / "other"
    other.mkdir()
    init_project(other, "other")
    (other / ".lobster_workspace" / ".env").write_text("LOBSTER_LAMIN=1\n")
    assert isinstance(
        get_catalog(run_workspace(project, "a"), environ={}), ThreadedCatalog
    )
    second = get_catalog(run_workspace(other, "a"), environ={})
    assert (
        isinstance(second, NullCatalog) and "already records" in second.status().reason
    )


def test_an_old_project_catalog_is_not_used_for_a_run_and_the_reason_names_the_fix(
    project, engines
):
    switch_on(project)
    made_before(project)
    catalog = get_catalog(run_workspace(project, "a"), environ={})
    assert isinstance(catalog, NullCatalog)
    reason = catalog.status().reason
    assert (
        "mv " in reason and str(project / ".lobster_workspace" / ".lamindb") in reason
    )
    assert engines == []


def test_an_old_project_catalog_still_serves_the_projects_own_workspace(
    project, engines
):
    switch_on(project)
    made_before(project)
    catalog = get_catalog(project / ".lobster_workspace", environ={})
    assert isinstance(catalog, ThreadedCatalog)
    assert catalog.key_root == project / ".lobster_workspace"


# -- nothing happens while the catalog is off -----------------------------------


def test_with_the_catalog_off_runs_leave_no_catalog_and_load_no_lamindb(project):
    code = (
        "import sys, json\n"
        "from pathlib import Path\n"
        "from lobster.core.catalog.factory import get_catalog\n"
        "p = Path(sys.argv[1])\n"
        "kinds = []\n"
        "for name in ('a', 'b'):\n"
        "    ws = p / 'runs' / name / '.lobster_workspace'\n"
        "    ws.mkdir(parents=True)\n"
        "    kinds.append(type(get_catalog(ws)).__name__)\n"
        "print(json.dumps({'kinds': kinds, 'lamin': sorted(\n"
        "    m for m in sys.modules if m.split('.')[0] in ('lamindb', 'lamindb_setup', 'django'))}))\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOBSTER_LAMIN")}
    env["HOME"] = str(project.parent / "home")
    done = subprocess.run(
        [sys.executable, "-c", code, str(project)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {"kinds": ["NullCatalog"] * 2, "lamin": []}
    found = {p.name for p in project.rglob("*")}
    assert ".lamindb" not in found and "catalog.lock" not in found


# -- what a save is recorded as -------------------------------------------------


def adata(seed):
    import anndata as ad
    import numpy as np

    rng = np.random.default_rng(seed)
    a = ad.AnnData(rng.poisson(1.0, size=(20, 5)).astype(np.float32))
    a.obs_names = [f"c{i}" for i in range(20)]
    a.var_names = [f"g{i}" for i in range(5)]
    return a


def save_in(workspace, session, seed=1):
    from lobster.core.runtime.data_manager import DataManagerV2

    dm = DataManagerV2(workspace_path=workspace)
    catalog = factory.begin_session_run(dm, workspace, session)
    dm.store_modality("x", adata(seed))
    dm.save_modality("x", "x.h5ad")
    return catalog


def registered_keys(engine):
    return [c[1]["key"] for c in engine.calls if c[0] == "register_artifact"]


def test_two_runs_saving_the_same_name_are_two_files_in_one_catalog(project, engines):
    switch_on(project)
    first = save_in(run_workspace(project, "a"), "s1", seed=1)
    second = save_in(run_workspace(project, "b"), "s2", seed=2)
    first.close("completed")
    assert second is first and len(engines) == 1
    assert registered_keys(engines[0]) == [
        "runs/a/.lobster_workspace/data/x.h5ad",
        "runs/b/.lobster_workspace/data/x.h5ad",
    ]
    begun = [c[1].session_id for c in engines[0].calls if c[0] == "begin_run"]
    assert begun == ["s1", "s2"]


def test_a_plain_workspace_saves_are_keyed_as_before(tmp_path, engines, monkeypatch):
    monkeypatch.setenv("LOBSTER_LAMIN", "1")
    ws = tmp_path / "ws"
    ws.mkdir()
    catalog = save_in(ws, "s1")
    catalog.close("completed")
    assert registered_keys(engines[0]) == ["data/x.h5ad"]


def test_the_projects_own_workspace_is_keyed_from_the_project(project, engines):
    switch_on(project)
    catalog = save_in(project / ".lobster_workspace", "s1")
    catalog.close("completed")
    assert registered_keys(engines[0]) == [".lobster_workspace/data/x.h5ad"]


def test_runs_that_overlap_share_the_one_worker_and_both_are_recorded(project, engines):
    switch_on(project)
    workspaces = [run_workspace(project, n) for n in ("a", "b")]
    gate = threading.Barrier(2)

    def go(ws, seed):
        gate.wait()
        save_in(ws, f"s{seed}", seed=seed)

    threads = [
        threading.Thread(target=go, args=(w, i)) for i, w in enumerate(workspaces)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    catalog = get_catalog(workspaces[0], environ={})
    catalog.close("completed")
    assert sorted(registered_keys(engines[0])) == [
        "runs/a/.lobster_workspace/data/x.h5ad",
        "runs/b/.lobster_workspace/data/x.h5ad",
    ]
    assert len(set(engines[0].threads)) == 1


# -- the commands say where the records are -------------------------------------


@pytest.fixture
def lamin(monkeypatch):
    from typer.testing import CliRunner

    from lobster.cli import app

    monkeypatch.delenv("LOBSTER_LAMIN", raising=False)
    monkeypatch.delenv("LOBSTER_LAMIN_INSTANCE", raising=False)

    def invoke(*args):
        result = CliRunner().invoke(app, ["lamin", *args])
        return result.exit_code, " ".join(result.output.split())

    return invoke


def test_status_from_a_run_names_the_projects_catalog(project, lamin):
    switch_on(project)
    code, out = lamin("status", "-w", str(run_workspace(project, "a")))
    assert "Catalog: on (setting from project)" in out
    assert f"catalog of this project in {project / '.lamindb'}" in out
    assert "not recorded" not in out


def test_status_does_not_say_on_alone_when_the_runs_are_not_recorded(project, lamin):
    switch_on(project)
    old = made_before(project)
    for ws in (project / ".lobster_workspace", run_workspace(project, "a")):
        code, out = lamin("status", "-w", str(ws))
        assert "Catalog: on" in out
        assert "Runs of this project are not recorded" in out
        assert f"mv {old} {old}.before-runs" in out


def test_status_with_the_catalog_off_has_nothing_to_add(project, lamin):
    code, out = lamin("status", "-w", str(run_workspace(project, "a")))
    assert "Catalog: off" in out and "not recorded" not in out


def test_a_run_folder_for_w_is_its_nested_workspace(project, lamin):
    switch_on(project)
    nested = run_workspace(project, "a")
    seen = {w: lamin("status", "-w", str(w)) for w in (nested, nested.parent)}
    assert seen[nested.parent] == seen[nested]
    assert "Catalog: on (setting from project)" in seen[nested][1]
    assert "Catalog: off" not in seen[nested][1]


@pytest.mark.parametrize("flags", [[], ["--runs"]])
def test_ls_of_a_run_folder_opens_the_nested_workspace(
    project, lamin, monkeypatch, flags
):
    from lobster.cli_internal.commands.light import lamin_commands
    from lobster.core.catalog.lamin import CatalogUnavailable

    opened = []

    def refuse(workspace, **kwargs):
        opened.append(workspace)
        raise CatalogUnavailable("stop here")

    folder = project / ".lamindb"
    folder.mkdir()
    (folder / "lobster.json").write_text(
        json.dumps({"slug": "anonymous/x", "created_by": "lobster"})
    )
    nested = run_workspace(project, "a")
    monkeypatch.setattr(lamin_commands, "_engine_installed", lambda: True)
    monkeypatch.setattr(lamin_commands, "_open", refuse)
    code, out = lamin("ls", *flags, "-w", str(nested.parent))
    assert opened == [nested] and "stop here" in out


def test_the_notebook_a_run_exports_is_keyed_from_the_project(project):
    from lobster.cli_internal.commands.light.replay_commands import _notebook_key

    run = run_workspace(project, "a")
    notebook = run.parent / "lobster_run.ipynb"
    assert _notebook_key(notebook, run) == "runs/a/lobster_run.ipynb"


# -- deleting a project removes its own catalog, and only that -------------------


def own_catalog(project):
    folder = project / ".lamindb"
    folder.mkdir()
    slug = "anonymous/lobster-study-1a2b3c"
    (folder / "lobster.json").write_text(
        json.dumps({"slug": slug, "created_by": "lobster"})
    )
    (folder / "storage_uid.txt").write_text(
        f"x\ncreation info:\ninstance_slug={slug}\n"
    )
    return folder


def test_delete_plans_the_projects_own_catalog(project):
    from lobster.core.projects import plan_delete

    folder = own_catalog(project)
    plan = plan_delete(project)
    assert folder in plan.paths and ".lamindb" not in plan.kept


def test_delete_keeps_a_lamindb_folder_that_is_not_the_projects_catalog(project):
    from lobster.core.projects import plan_delete

    foreign = project / ".lamindb"
    foreign.mkdir()
    (foreign / "storage_uid.txt").write_text("instance_slug=me/mine\n")
    plan = plan_delete(project)
    assert foreign not in plan.paths and ".lamindb" in plan.kept


def test_delete_keeps_a_catalog_whose_two_marks_disagree(project):
    from lobster.core.projects import plan_delete

    folder = own_catalog(project)
    (folder / "storage_uid.txt").write_text("instance_slug=anonymous/other\n")
    assert folder not in plan_delete(project).paths


def test_delete_lists_the_catalog_in_a_dry_run_and_removes_it(project):
    from typer.testing import CliRunner

    from lobster.cli import app

    folder = own_catalog(project)
    dry = CliRunner().invoke(app, ["projects", "delete", str(project), "--dry-run"])
    assert dry.exit_code == 0 and str(folder) in dry.output.replace("\n", "")
    assert folder.exists()
    done = CliRunner().invoke(app, ["projects", "delete", str(project), "--yes"])
    assert done.exit_code == 0, done.output
    assert not folder.exists() and not (project / ".lobster_workspace").exists()
