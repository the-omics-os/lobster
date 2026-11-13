"""ThreadedCatalog: one worker thread, inter-process write lock, fail-open."""

import contextlib
import os
import threading
import time
from pathlib import Path

import pytest

from lobster.core.catalog import thread as catalog_thread
from lobster.core.catalog.port import Catalog, SessionInfo
from lobster.core.catalog.thread import ThreadedCatalog

pytestmark = pytest.mark.unit


def _session(tmp_path, sid="session_20261108_100000"):
    return SessionInfo(
        session_id=sid,
        workspace_path=tmp_path,
        lobster_version="0.0.0",
        session_dir=tmp_path / ".lobster" / "sessions" / sid,
    )


@pytest.fixture
def engine(recording_catalog_cls):
    return recording_catalog_cls()


@pytest.fixture
def make_catalog(tmp_path, engine):
    made = []

    def make(factory=None, timeout=5.0):
        catalog = ThreadedCatalog(
            factory or (lambda: engine),
            lock_path=tmp_path / ".lobster" / "catalog.lock",
            engine="fake",
            mode="workspace",
            timeout=timeout,
        )
        made.append(catalog)
        return catalog

    yield make
    for catalog in made:
        for event in engine.block.values():
            event.set()
        catalog.close()


def test_satisfies_the_port(make_catalog):
    assert isinstance(make_catalog(), Catalog)


def test_engine_built_and_called_on_one_worker_thread(make_catalog, engine, tmp_path):
    built_on = []

    def factory():
        built_on.append(threading.get_ident())
        return engine

    catalog = make_catalog(factory)
    catalog.begin_run(_session(tmp_path))
    results = []

    def save(i):
        results.append(
            catalog.register_artifact(
                tmp_path / f"data/x{i}.h5ad",
                key=f"data/x{i}.h5ad",
                kind="dataset",
                otype="AnnData",
                description="",
                meta={},
            )
        )

    callers = [threading.Thread(target=save, args=(i,)) for i in range(8)]
    for t in callers:
        t.start()
    for t in callers:
        t.join()

    assert len(results) == 8 and all(r is not None for r in results)
    worker = catalog.worker_ident
    assert worker not in {t.ident for t in callers} | {threading.get_ident()}
    assert set(engine.threads) == {worker} and built_on == [worker]
    assert engine.ops()[0] == "begin_run"


def test_begin_run_is_fire_and_forget(make_catalog, engine, tmp_path):
    engine.block["begin_run"] = threading.Event()
    catalog = make_catalog()
    started = time.monotonic()
    catalog.begin_run(_session(tmp_path))
    assert time.monotonic() - started < 1.0
    engine.block["begin_run"].set()
    # The next call waits behind it on the worker.
    assert catalog.keep_path(tmp_path / "data/x.h5ad") is None
    assert engine.ops()[:2] == ["begin_run", "keep_path"]


def test_writes_hold_the_workspace_lock_status_does_not(
    make_catalog, engine, tmp_path, monkeypatch
):
    held = []

    @contextlib.contextmanager
    def spy_lock(path, wait=None, abandoned=None):
        held.append(Path(path))
        yield

    monkeypatch.setattr(catalog_thread, "workspace_lock", spy_lock)
    catalog = make_catalog()
    lock = tmp_path / ".lobster" / "catalog.lock"

    catalog.status()
    assert held == []

    catalog.keep_path(tmp_path / "data/x.h5ad")  # decides against the catalog: a write
    assert held == [lock]

    catalog.begin_run(_session(tmp_path))
    catalog.register_artifact(
        tmp_path / "data/x.h5ad",
        key="data/x.h5ad",
        kind="dataset",
        otype=None,
        description="",
        meta={},
    )
    catalog.link_input(tmp_path / "data/x.h5ad")
    catalog.register_transform(
        key="notebooks/n.ipynb", kind="notebook", source="{}", description=""
    )
    catalog.end_run("completed")
    assert held == [lock] * 6


def test_real_lock_file_is_created_in_lobster_dir(make_catalog, tmp_path):
    catalog = make_catalog()
    catalog.link_input(tmp_path / "x")
    catalog.status()  # queued behind link_input: it has run
    assert (tmp_path / ".lobster" / "catalog.lock").exists()


def test_engine_error_returns_none_and_is_reported_once(
    make_catalog, engine, tmp_path, caplog
):
    engine.raise_on["register_artifact"] = RuntimeError("boom")
    catalog = make_catalog()
    kwargs = dict(
        key="data/x.h5ad", kind="dataset", otype=None, description="", meta={}
    )
    with caplog.at_level("WARNING", logger="lobster.core.catalog.thread"):
        assert catalog.register_artifact(tmp_path / "data/x.h5ad", **kwargs) is None
        assert catalog.register_artifact(tmp_path / "data/x.h5ad", **kwargs) is None
    assert (
        sum("register_artifact failed" in r.getMessage() for r in caplog.records) == 1
    )
    assert not catalog.degraded
    # Other calls keep working.
    assert catalog.link_input(tmp_path / "data/x.h5ad") is None
    catalog.status()
    assert engine.ops().count("link_input") == 1


def test_system_exit_from_engine_does_not_escape_or_kill_the_worker(
    make_catalog, engine, tmp_path
):
    engine.raise_on["link_input"] = SystemExit("engine said exit")
    catalog = make_catalog()
    assert catalog.link_input(tmp_path / "x") is None
    assert catalog.keep_path(tmp_path / "y.h5ad") is None
    assert engine.ops()[-1] == "keep_path"


def test_timeout_degrades_and_skips_later_calls(make_catalog, engine, tmp_path):
    engine.block["register_transform"] = threading.Event()
    catalog = make_catalog(timeout=0.2)
    started = time.monotonic()
    ref = catalog.register_transform(
        key="notebooks/n.ipynb", kind="notebook", source="{}", description=""
    )
    assert ref is None and time.monotonic() - started < 2.0
    assert catalog.degraded
    status = catalog.status()
    assert status.degraded and "register_transform" in status.reason
    # Skipped without touching the (stuck) engine.
    target = tmp_path / "data/y.h5ad"
    assert catalog.keep_path(target) is None
    assert catalog.link_input(target) is None
    engine.block["register_transform"].set()
    assert "keep_path" not in engine.ops() and "link_input" not in engine.ops()


def test_a_slow_registration_finishes_in_the_background(
    make_catalog, engine, tmp_path, monkeypatch
):
    monkeypatch.setattr(catalog_thread, "REGISTER_WAIT_S", 0.2)
    engine.block["register_artifact"] = threading.Event()
    catalog = make_catalog()
    started = time.monotonic()
    ref = catalog.register_artifact(
        tmp_path / "data/x.h5ad",
        key="data/x.h5ad",
        kind="dataset",
        otype=None,
        description="",
        meta={},
    )
    assert ref is None and time.monotonic() - started < 1.0
    assert not catalog.degraded
    engine.block["register_artifact"].set()
    catalog.status()
    assert engine.ops() == ["register_artifact", "status"]


def test_engine_that_cannot_start_degrades(make_catalog, tmp_path):
    def broken():
        raise ImportError("no engine")

    catalog = make_catalog(broken)
    catalog.begin_run(_session(tmp_path))
    target = tmp_path / "data/x.h5ad"
    assert catalog.keep_path(target) is None
    assert catalog.degraded
    assert "could not start" in catalog.status().reason


@pytest.mark.parametrize("answer", ["same", "existing", None, 42])
def test_unusable_keep_answers_are_dropped(make_catalog, engine, tmp_path, answer):
    target = tmp_path / "data" / "x.h5ad"
    existing = tmp_path / "taken.h5ad"
    existing.write_text("mine")
    engine.next_keep_path = {"same": target, "existing": existing}.get(answer, answer)
    assert make_catalog().keep_path(target) is None


def test_keep_answer_is_passed_on(make_catalog, engine, tmp_path):
    target = tmp_path / "data" / "x.h5ad"
    engine.next_keep_path = tmp_path / ".lamindb" / "versions" / "v1.h5ad"
    assert make_catalog().keep_path(target) == engine.next_keep_path


def test_a_late_keep_answer_is_skipped_without_degrading(
    make_catalog, engine, tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(catalog_thread, "KEEP_TIMEOUT_S", 0.2)
    engine.block["keep_path"] = threading.Event()
    engine.next_keep_path = tmp_path / "kept.h5ad"
    catalog = make_catalog()
    t0 = time.monotonic()
    with caplog.at_level("WARNING", logger="lobster.core.catalog.thread"):
        assert catalog.keep_path(tmp_path / "x.h5ad") is None
    assert time.monotonic() - t0 < 1.0
    assert not catalog.degraded
    assert any("did not finish within 0.2 s" in r.getMessage() for r in caplog.records)
    engine.block["keep_path"].set()
    assert catalog.link_input(tmp_path / "x.h5ad") is None  # the worker goes on
    catalog.status()
    assert engine.ops() == ["keep_path", "link_input", "status"]


def test_a_call_that_timed_out_while_queued_never_runs(
    make_catalog, engine, tmp_path, monkeypatch
):
    monkeypatch.setattr(catalog_thread, "KEEP_TIMEOUT_S", 0.2)
    engine.block["begin_run"] = threading.Event()
    catalog = make_catalog()
    catalog.begin_run(_session(tmp_path))  # holds the worker
    assert catalog.keep_path(tmp_path / "x.h5ad") is None
    engine.block["begin_run"].set()
    catalog.status()  # queued behind both
    assert engine.ops() == ["begin_run", "status"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock")
def test_another_process_holding_the_lock_is_waited_for_briefly(
    make_catalog, engine, tmp_path, monkeypatch, caplog
):
    import fcntl

    monkeypatch.setattr(catalog_thread, "LOCK_WAIT_S", 0.2)
    catalog = make_catalog()
    catalog.status()  # started
    lock = tmp_path / ".lobster" / "catalog.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a+") as other:  # another open file: what another process holds
        fcntl.flock(other.fileno(), fcntl.LOCK_EX)
        t0 = time.monotonic()
        with caplog.at_level("WARNING", logger="lobster.core.catalog.thread"):
            assert catalog.keep_path(tmp_path / "x.h5ad") is None
        assert time.monotonic() - t0 < 1.5
        fcntl.flock(other.fileno(), fcntl.LOCK_UN)
    assert any("held the catalog lock" in r.getMessage() for r in caplog.records)
    assert "keep_path" not in engine.ops() and not catalog.degraded
    assert catalog.keep_path(tmp_path / "x.h5ad") is None
    assert engine.ops()[-1] == "keep_path"


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock")
def test_a_write_abandoned_while_waiting_for_the_lock_never_runs(
    make_catalog, engine, tmp_path
):
    import fcntl

    catalog = make_catalog(timeout=0.3)
    catalog.status()
    lock = tmp_path / ".lobster" / "catalog.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a+") as other:
        fcntl.flock(other.fileno(), fcntl.LOCK_EX)
        ref = catalog.register_transform(
            key="notebooks/n.ipynb", kind="notebook", source="{}", description=""
        )
        assert ref is None and catalog.degraded
        time.sleep(0.2)  # the worker notices it was abandoned
        fcntl.flock(other.fileno(), fcntl.LOCK_UN)
    time.sleep(0.2)
    assert "register_transform" not in engine.ops()


def test_status_comes_from_the_engine(make_catalog):
    status = make_catalog().status()
    assert status.engine == "fake" and status.location == "here" and not status.degraded


def test_exit_ends_an_open_run_with_the_session_report(make_catalog, engine, tmp_path):
    session = _session(tmp_path)
    session.session_dir.mkdir(parents=True)
    (session.session_dir / "provenance.jsonl").write_text("{}\n")
    catalog = make_catalog()
    catalog.begin_run(session)
    catalog._at_exit()
    assert engine.calls[-1] == (
        "end_run",
        ("completed", session.session_dir / "provenance.jsonl"),
    )


def test_a_session_switch_ends_the_previous_run_with_its_own_report(
    make_catalog, engine, tmp_path
):
    first, second = _session(tmp_path, "s1"), _session(tmp_path, "s2")
    for session in (first, second):
        session.session_dir.mkdir(parents=True)
        (session.session_dir / "provenance.jsonl").write_text("{}\n")
    catalog = make_catalog()
    catalog.begin_run(first)
    catalog.begin_run(second)
    catalog.status()  # the worker has run everything queued before
    runs = [c for c in engine.calls if c[0] in ("begin_run", "end_run")]
    assert runs == [
        ("begin_run", first),
        ("end_run", ("completed", first.session_dir / "provenance.jsonl")),
        ("begin_run", second),
    ]
    catalog._at_exit()
    assert engine.calls[-1] == (
        "end_run",
        ("completed", second.session_dir / "provenance.jsonl"),
    )


def test_concurrent_switches_keep_each_end_with_its_own_begin(
    make_catalog, engine, tmp_path
):
    sessions = [_session(tmp_path, f"s{i}") for i in range(3)]
    for session in sessions:
        session.session_dir.mkdir(parents=True)
        (session.session_dir / "provenance.jsonl").write_text("{}\n")
    catalog = make_catalog()
    catalog.begin_run(sessions[0])
    entered = threading.Event()
    queue_call = catalog._call

    def slow(op, *args, **kwargs):
        if op == "end_run" and not entered.is_set():
            entered.set()
            time.sleep(0.3)  # the other switch gets its chance in here
        return queue_call(op, *args, **kwargs)

    catalog._call = slow
    first = threading.Thread(target=catalog.begin_run, args=(sessions[1],))
    first.start()
    assert entered.wait(5)
    catalog.begin_run(sessions[2])
    first.join(5)
    catalog.status()
    logs = [s.session_dir / "provenance.jsonl" for s in sessions]
    assert [c for c in engine.calls if c[0] in ("begin_run", "end_run")] == [
        ("begin_run", sessions[0]),
        ("end_run", ("completed", logs[0])),
        ("begin_run", sessions[1]),
        ("end_run", ("completed", logs[1])),
        ("begin_run", sessions[2]),
    ]


def test_a_failed_end_of_the_left_run_does_not_deadlock_the_switch(
    make_catalog, engine, tmp_path
):
    """A queued call's done-callback may run on the submitting thread, inside
    ``begin_run``'s hold of the state lock; reporting the failure takes the lock
    again, so it must be reentrant."""
    engine.raise_on["end_run"] = RuntimeError("boom")
    catalog = make_catalog()
    catalog.begin_run(_session(tmp_path, "s1"))
    # The engine is published under the state lock the patched submit holds below.
    assert catalog._ready.wait(5)
    submit = catalog._worker.submit

    def failed_before_the_callback_is_added(body):
        future = submit(body)
        future.exception(timeout=5)
        return future

    catalog._worker.submit = failed_before_the_callback_is_added
    switch = threading.Thread(
        target=catalog.begin_run, args=(_session(tmp_path, "s2"),), daemon=True
    )
    switch.start()
    switch.join(5)
    stuck = switch.is_alive()
    if stuck:  # a plain Lock: free it, or the fixture's teardown would hang too
        catalog._state_lock.release()
        switch.join(5)
    assert not stuck


def test_exit_without_open_run_records_nothing(make_catalog, engine, tmp_path):
    catalog = make_catalog()
    catalog.begin_run(_session(tmp_path))
    catalog.end_run("errored")
    catalog._at_exit()
    assert engine.ops().count("end_run") == 1


def test_the_catalog_lock_excludes_the_cli_lock(tmp_path):
    from lobster.core.queues.queue_storage import InterProcessFileLock

    lock = tmp_path / ".lobster" / "catalog.lock"
    with InterProcessFileLock(lock):
        with pytest.raises(catalog_thread._LockBusy):
            with catalog_thread.workspace_lock(lock, wait=0.1):
                pass
    with catalog_thread.workspace_lock(lock, wait=0.1):
        pass


def _other_process_can_lock(lock):
    import fcntl

    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a+") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return True


_FIELDS = dict(key="data/x.h5ad", kind="dataset", otype=None, description="", meta={})


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock")
def test_a_save_section_holds_the_lock_until_the_file_is_registered(
    make_catalog, engine, tmp_path
):
    catalog = make_catalog()
    catalog.status()
    lock = tmp_path / ".lobster" / "catalog.lock"
    target = tmp_path / "data" / "x.h5ad"
    target.parent.mkdir()
    target.write_text("1")
    with catalog.save_section(target) as save:
        assert save.recorded and save.keep is None
        assert not _other_process_can_lock(lock)
        target.write_text("2")
        catalog.link_input(target)  # this process meanwhile: runs, no deadlock
        ref = save.register(**_FIELDS)
        assert ref is not None and ref.path == target
    catalog.status()
    assert _other_process_can_lock(lock)
    assert engine.ops() == [
        "status",
        "keep_path",
        "link_input",
        "register_artifact",
        "status",
    ]


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock")
def test_a_failed_write_ends_the_section_without_a_record(
    make_catalog, engine, tmp_path
):
    catalog = make_catalog()
    target = tmp_path / "x.h5ad"
    with pytest.raises(RuntimeError):
        with catalog.save_section(target):
            raise RuntimeError("disk full")
    catalog.status()
    assert _other_process_can_lock(tmp_path / ".lobster" / "catalog.lock")
    assert "register_artifact" not in engine.ops()


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock")
def test_a_busy_lock_leaves_the_save_unrecorded_and_marks_later(
    make_catalog, engine, tmp_path, monkeypatch
):
    import fcntl

    monkeypatch.setattr(catalog_thread, "LOCK_WAIT_S", 0.2)
    catalog = make_catalog()
    catalog.status()
    lock = tmp_path / ".lobster" / "catalog.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "x.h5ad"
    with open(lock, "a+") as other:
        fcntl.flock(other.fileno(), fcntl.LOCK_EX)
        t0 = time.monotonic()
        with catalog.save_section(target) as save:
            assert not save.recorded and save.keep is None
            assert save.register(**_FIELDS) is None
        assert time.monotonic() - t0 < 1.5
        fcntl.flock(other.fileno(), fcntl.LOCK_UN)
    catalog.status()
    assert not catalog.degraded
    assert engine.ops() == ["status", "note_unrecorded", "status"]


def test_a_file_replaced_before_its_registration_is_not_recorded(
    make_catalog, engine, tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(catalog_thread, "REGISTER_WAIT_S", 0.2)
    catalog = make_catalog()
    target = tmp_path / "x.h5ad"
    with catalog.save_section(target) as save:
        target.write_text("ours")
        engine.block["link_input"] = threading.Event()
        catalog.link_input(target)  # keeps the worker busy
        assert save.register(**_FIELDS) is None  # still queued
        target.write_text("someone else's, longer")
        engine.block["link_input"].set()
    with caplog.at_level("WARNING", logger="lobster.core.catalog.thread"):
        catalog.status()
    assert "register_artifact" not in engine.ops()
    assert any("replaced by another writer" in r.getMessage() for r in caplog.records)


def test_a_section_while_starting_registers_later_unprepared(
    make_catalog, engine, tmp_path
):
    started = threading.Event()

    def slow():
        started.wait(10)
        return engine

    catalog = make_catalog(slow)
    catalog._startup_wait = 0.1
    target = tmp_path / "x.h5ad"
    with catalog.save_section(target) as save:
        assert save.recorded and save.keep is None
        target.write_text("1")
        assert save.register(**_FIELDS) is None  # queued behind the start
    started.set()
    catalog.status()
    assert engine.ops() == ["register_artifact", "status"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock")
def test_a_late_keep_lets_the_lock_go(make_catalog, engine, tmp_path, monkeypatch):
    monkeypatch.setattr(catalog_thread, "KEEP_TIMEOUT_S", 0.2)
    engine.block["keep_path"] = threading.Event()
    catalog = make_catalog()
    catalog.status()
    lock = tmp_path / ".lobster" / "catalog.lock"
    target = tmp_path / "x.h5ad"
    with catalog.save_section(target) as save:
        assert not save.recorded
        assert not _other_process_can_lock(lock)  # the stalled keep holds it
    engine.block["keep_path"].set()
    catalog.status()  # the stalled section has ended on the worker
    assert _other_process_can_lock(lock)
    assert not catalog.degraded
    assert "register_artifact" not in engine.ops()


def test_a_file_changed_while_it_is_recorded_is_marked(
    make_catalog, engine, tmp_path, caplog
):
    catalog = make_catalog()
    target = tmp_path / "x.h5ad"
    original = engine.register_artifact

    def register_while_someone_writes(path, **fields):
        ref = original(path, **fields)
        target.write_text("someone else's, longer")
        return ref

    engine.register_artifact = register_while_someone_writes
    with catalog.save_section(target) as save:
        target.write_text("ours")
        with caplog.at_level("WARNING", logger="lobster.core.catalog.thread"):
            assert save.register(**_FIELDS) is not None
    catalog.status()
    assert engine.ops()[-3:] == ["register_artifact", "note_unrecorded", "status"]
    assert any("while it was recorded" in r.getMessage() for r in caplog.records)


def test_different_problems_of_one_operation_are_each_reported_once(
    make_catalog, caplog
):
    catalog = make_catalog()
    with caplog.at_level("WARNING", logger="lobster.core.catalog.thread"):
        for detail in ("first problem", "first problem", "second problem"):
            catalog._report("save", detail)
    assert [r.getMessage().split("(")[-1] for r in caplog.records] == [
        "first problem)",
        "second problem)",
    ]
