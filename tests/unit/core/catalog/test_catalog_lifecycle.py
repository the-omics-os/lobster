"""Start-up budget, fork, exit and concurrency behaviour of the catalog seam."""

import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pytest

from lobster.core.catalog import factory
from lobster.core.catalog import thread as catalog_thread
from lobster.core.catalog.null import NullCatalog
from lobster.core.catalog.port import SessionInfo
from lobster.core.catalog.thread import ThreadedCatalog
from lobster.core.runtime.data_manager import DataManagerV2

pytestmark = pytest.mark.unit


def _adata(seed=0):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(rng.poisson(1.0, size=(20, 8)).astype(np.float32))
    a.obs_names = [f"c{i}" for i in range(20)]
    a.var_names = [f"g{i}" for i in range(8)]
    return a


def _session(ws):
    return SessionInfo(session_id="session_x", workspace_path=ws, lobster_version="0")


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    path.mkdir()
    return path


@pytest.fixture
def engine(recording_catalog_cls):
    return recording_catalog_cls()


@pytest.fixture
def make_catalog(ws, engine):
    made = []

    def make(engine_factory=None, **kw):
        kw.setdefault("timeout", 5.0)
        catalog = ThreadedCatalog(
            engine_factory or (lambda: engine),
            lock_path=ws / ".lobster" / "catalog.lock",
            engine="fake",
            mode="workspace",
            **kw,
        )
        made.append(catalog)
        return catalog

    yield make
    for event in engine.block.values():
        event.set()
    for catalog in made:
        catalog.close()


# -- start-up budget -----------------------------------------------------------------------


def test_slow_start_neither_stalls_the_first_save_nor_degrades(
    make_catalog, engine, ws
):
    started = threading.Event()

    def slow_factory():
        started.wait(10)
        return engine

    catalog = make_catalog(slow_factory, startup_wait=0.2, timeout=0.3)
    catalog.begin_run(_session(ws))
    target = ws / "data" / "x.h5ad"
    t0 = time.monotonic()
    assert catalog.keep_path(target) is None  # not prepared while starting
    assert (
        catalog.register_artifact(
            target,
            key="data/x.h5ad",
            kind="dataset",
            otype=None,
            description="",
            meta={},
        )
        is None
    )  # write: queued behind the build
    assert time.monotonic() - t0 < 1.5
    assert not catalog.degraded

    started.set()
    assert catalog.keep_path(target) is None  # waits for the engine now
    assert engine.ops() == ["begin_run", "register_artifact", "keep_path"]
    assert not catalog.degraded


def test_start_beyond_its_own_budget_degrades(make_catalog, engine):
    gate = threading.Event()

    def stuck_factory():
        gate.wait(10)
        return engine

    catalog = make_catalog(stuck_factory, startup_wait=0.05, build_timeout=0.3)
    time.sleep(0.6)
    assert catalog.degraded and "did not start within" in catalog.status().reason
    gate.set()


def test_calls_queued_behind_an_over_budget_start_never_run(make_catalog, engine, ws):
    gate = threading.Event()

    def stuck_factory():
        gate.wait(10)
        return engine

    catalog = make_catalog(stuck_factory, startup_wait=0.05, build_timeout=0.3)
    catalog.begin_run(_session(ws))  # queued behind the build
    target = ws / "data" / "x.h5ad"
    catalog.register_artifact(
        target, key="data/x.h5ad", kind="dataset", otype=None, description="", meta={}
    )
    time.sleep(0.6)
    assert catalog.degraded
    gate.set()
    catalog._worker.submit(lambda: None).result(5)  # the queue has drained
    assert engine.ops() == []


def test_watchdog_latching_during_the_build_keeps_the_engine_unused(
    make_catalog, engine, ws
):
    holder = {}
    assigned = threading.Event()

    def factory_racing_the_watchdog():
        # The watchdog fires after the engine exists but before it is published.
        assert assigned.wait(5)
        holder["catalog"]._check_started()
        return engine

    holder["catalog"] = catalog = make_catalog(
        factory_racing_the_watchdog, startup_wait=0.05
    )
    assigned.set()
    catalog.begin_run(_session(ws))
    catalog._worker.submit(lambda: None).result(5)
    assert catalog.degraded and engine.ops() == []
    assert "did not start within" in catalog.status().reason
    assert catalog._engine is None


def test_watchdog_after_the_build_stands_down(make_catalog, engine, ws):
    catalog = make_catalog()
    catalog.status()
    catalog._check_started()
    assert not catalog.degraded


def test_no_run_is_ever_half_recorded_around_the_start_budget(engine, ws):
    """Build time straddles the budget: each catalog either records or latches."""
    for i in range(40):
        engine.calls.clear()

        def factory_near_budget():
            time.sleep(0.02)
            return engine

        catalog = ThreadedCatalog(
            factory_near_budget,
            lock_path=ws / ".lobster" / "catalog.lock",
            engine="fake",
            mode="workspace",
            startup_wait=0.0,
            build_timeout=0.02,
            timeout=5.0,
        )
        catalog.begin_run(_session(ws))
        catalog._worker.submit(lambda: None).result(5)
        ops = engine.ops()
        assert (catalog.degraded and ops == []) or (
            not catalog.degraded and ops == ["begin_run"]
        ), (i, catalog.status().reason, ops)
        catalog.close()


def test_build_failure_reason_carries_the_message(make_catalog):
    def failing_factory():
        raise ValueError("settings are missing from .lamindb/settings")

    catalog = make_catalog(failing_factory)
    reason = catalog.status().reason
    assert "ValueError: settings are missing" in reason and len(reason) < 260


# -- link_input never blocks ----------------------------------------------------------------


def test_link_input_does_not_wait_for_the_engine(make_catalog, engine, ws):
    engine.block["link_input"] = threading.Event()
    catalog = make_catalog()
    catalog.status()  # engine built
    t0 = time.monotonic()
    assert catalog.link_input(ws / "big.h5ad") is None
    assert time.monotonic() - t0 < 0.5
    engine.block["link_input"].set()


# -- degraded / closed: saves go on, without waiting ----------------------------------------


def test_save_after_degradation_returns_the_requested_path_without_waiting(
    make_catalog, engine, ws
):
    engine.block["status"] = threading.Event()
    catalog = make_catalog(timeout=0.2)
    dm = DataManagerV2(workspace_path=ws)
    dm.attach_catalog(catalog)
    dm.store_modality("x", _adata())
    catalog.status()  # times out -> degraded
    assert catalog.degraded
    t0 = time.monotonic()
    out = dm.save_modality("x", "x.h5ad")
    assert time.monotonic() - t0 < 1.0
    assert Path(out) == dm.data_dir / "x.h5ad" and Path(out).is_file()
    engine.block["status"].set()


def test_a_late_keep_leaves_the_file_at_its_name(make_catalog, engine, ws, monkeypatch):
    """The engine answers after the caller stopped waiting: nothing is moved.

    That save goes ahead unrecorded; once the lock is free the engine is told,
    so that no record points at the bytes it replaced.
    """
    monkeypatch.setattr(catalog_thread, "KEEP_TIMEOUT_S", 0.2)
    catalog = make_catalog()
    dm = DataManagerV2(workspace_path=ws)
    dm.attach_catalog(catalog)
    dm.store_modality("x", _adata())
    first = Path(dm.save_modality("x", "x.h5ad"))
    kept = ws / ".lamindb" / "versions" / "v1.h5ad"
    engine.next_keep_path = kept
    engine.block["keep_path"] = threading.Event()
    # The engine answers after the caller gave up, while the save is writing.
    late = threading.Timer(0.4, engine.block["keep_path"].set)
    late.start()
    dm.store_modality("x", _adata(seed=2))
    out = Path(dm.save_modality("x", "x.h5ad"))
    late.join()
    catalog.status()  # everything queued has run
    assert out == first and out.is_file() and not kept.exists()
    assert not catalog.degraded
    assert engine.ops() == [
        "keep_path",
        "register_artifact",
        "keep_path",
        "note_unrecorded",
        "status",
    ]


def test_hooks_after_close_are_no_ops(make_catalog, engine, ws):
    catalog = make_catalog()
    catalog.begin_run(_session(ws))
    catalog.close()
    n = len(engine.calls)
    dm = DataManagerV2(workspace_path=ws)
    dm.attach_catalog(catalog)
    dm.store_modality("x", _adata())
    assert Path(dm.save_modality("x", "x.h5ad")) == dm.data_dir / "x.h5ad"
    assert catalog.link_input(ws / "x") is None
    assert catalog.end_run("completed") is None
    assert len(engine.calls) == n


# -- concurrency through DataManagerV2 ------------------------------------------------------


def test_concurrent_saves_are_all_recorded_on_the_worker(make_catalog, engine, ws):
    catalog = make_catalog()
    dm = DataManagerV2(workspace_path=ws)
    dm.attach_catalog(catalog)
    names = [f"m{i}" for i in range(8)]
    for i, name in enumerate(names):
        dm.store_modality(name, _adata(i))
    errors = []

    def save(name):
        try:
            dm.save_modality(name, f"{name}.h5ad")
        except Exception as exc:  # pragma: no cover - failure detail
            errors.append(exc)

    threads = [threading.Thread(target=save, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    keys = sorted(p["key"] for op, p in engine.calls if op == "register_artifact")
    assert keys == sorted(f"data/{n}.h5ad" for n in names)
    assert set(engine.threads) == {catalog.worker_ident}


# -- exit and fork --------------------------------------------------------------------------


def test_atexit_registered_and_unregistered_on_close(ws, engine, monkeypatch):
    registered, unregistered = [], []
    monkeypatch.setattr(catalog_thread.atexit, "register", registered.append)
    monkeypatch.setattr(catalog_thread.atexit, "unregister", unregistered.append)
    catalog = ThreadedCatalog(
        lambda: engine,
        lock_path=ws / ".lobster" / "catalog.lock",
        engine="fake",
        mode="workspace",
    )
    assert registered == [catalog._at_exit]
    catalog.close()
    assert unregistered == [catalog._at_exit]


def test_worker_is_a_daemon(make_catalog):
    catalog = make_catalog()
    assert catalog._worker._thread.daemon


def test_exit_waits_for_an_engine_that_is_still_building(
    make_catalog, engine, ws, monkeypatch
):
    """A one-shot command can exit before the engine is built; its run is still recorded."""
    monkeypatch.setattr(catalog_thread, "AT_EXIT_TIMEOUT_S", 3.0)

    def slow_factory():
        time.sleep(0.8)
        return engine

    catalog = make_catalog(slow_factory, startup_wait=0.2, timeout=5.0)
    catalog.begin_run(_session(ws))
    catalog.set_end_status("completed")
    catalog._at_exit()
    assert engine.ops() == ["begin_run", "end_run"]
    assert not catalog.degraded


def test_exit_does_not_outlast_its_budget_for_a_build_that_never_ends(
    make_catalog, engine, ws, monkeypatch
):
    monkeypatch.setattr(catalog_thread, "AT_EXIT_TIMEOUT_S", 1.0)
    gate = threading.Event()

    def stuck_factory():
        gate.wait(10)
        return engine

    catalog = make_catalog(stuck_factory, startup_wait=0.2, timeout=5.0)
    catalog.begin_run(_session(ws))
    t0 = time.monotonic()
    catalog._at_exit()
    assert time.monotonic() - t0 < 1.6
    assert engine.ops() == []
    gate.set()


def test_fork_latch_degrades_and_skips_exit(make_catalog, engine, ws):
    catalog = make_catalog()
    catalog.begin_run(_session(ws))
    catalog.status()
    catalog_thread._after_fork_in_child()  # what os.register_at_fork runs in a child
    assert catalog.degraded and "forked" in catalog.status().reason
    n = len(engine.calls)
    catalog._at_exit()
    target = ws / "data" / "x.h5ad"
    assert catalog.keep_path(target) is None
    assert len(engine.calls) == n


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork only")
def test_real_fork_child_never_waits_on_the_dead_worker(make_catalog, engine, ws):
    catalog = make_catalog()
    catalog.status()
    pid = os.fork()
    if pid == 0:  # child: must answer at once, without a worker
        code = 1
        try:
            t0 = time.monotonic()
            ok = catalog.degraded and catalog.keep_path(ws / "a.h5ad") is None
            code = 0 if ok and time.monotonic() - t0 < 1.0 else 1
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    assert not catalog.degraded  # the parent keeps recording


# -- factory: no file reads when off ------------------------------------------------------


def test_explicit_off_reads_no_file(ws, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("settings file read")

    monkeypatch.setattr(factory, "_read_workspace_env", boom)
    monkeypatch.setattr(factory, "_read_global", boom)
    assert isinstance(
        factory.get_catalog(ws, environ={"LOBSTER_LAMIN": "0"}), NullCatalog
    )


def test_off_resolution_is_cached_per_process(ws, monkeypatch):
    monkeypatch.delenv("LOBSTER_LAMIN", raising=False)
    monkeypatch.delenv("LOBSTER_LAMIN_INSTANCE", raising=False)
    reads = []
    monkeypatch.setattr(factory, "_read_workspace_env", lambda p: reads.append(p) or {})
    monkeypatch.setattr(factory, "_read_global", lambda p: {})
    assert isinstance(factory.get_catalog(ws), NullCatalog)
    assert isinstance(factory.get_catalog(ws), NullCatalog)
    assert len(reads) == 1


# -- CommandClient: one run per invocation --------------------------------------------------


@pytest.mark.parametrize(
    "module", ["lobster.cli", "lobster.cli_internal.commands.heavy.session_infra"]
)
def test_command_client_begins_one_run(ws, engine, module):
    import importlib

    CommandClient = importlib.import_module(module).CommandClient
    with patch.object(factory, "get_catalog", return_value=engine):
        client = CommandClient(ws)
    assert client.data_manager.catalog is engine
    begins = [p for op, p in engine.calls if op == "begin_run"]
    assert len(begins) == 1 and begins[0].session_id.startswith("command_")
    assert begins[0].session_dir is None and begins[0].kind == "command"


def test_command_client_with_a_session_uses_its_id(ws, engine):
    from lobster.cli import CommandClient

    (ws / ".lobster" / "sessions" / "session_a").mkdir(parents=True)
    with patch.object(factory, "get_catalog", return_value=engine):
        CommandClient(ws, session_id="session_a")
    ((op, info),) = [c for c in engine.calls if c[0] == "begin_run"]
    assert info.session_id == "session_a" and info.kind == "command"
    assert info.session_dir == ws / ".lobster" / "sessions" / "session_a"


def test_command_client_survives_a_broken_catalog(ws):
    from lobster.cli import CommandClient

    with patch.object(factory, "get_catalog", side_effect=RuntimeError("boom")):
        client = CommandClient(ws)
    assert client.data_manager is not None
