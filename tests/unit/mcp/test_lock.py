"""One Lobster process at a time holds an output_dir's session folder.

The claim is a lock file inside the hidden workspace, created with
``O_CREAT | O_EXCL`` and naming the process that made it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from lobster.cli_internal import mcp_lock, mcp_record, mcp_turns


def lock_of(out: Path) -> Path:
    return out / mcp_record.WORKSPACE_DIR / mcp_lock.LOCK_NAME


def dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


@pytest.fixture
def foreign():
    """A live process that is not this one."""
    child = subprocess.Popen(["sleep", "60"])
    yield child.pid
    child.kill()
    child.wait()


def test_a_new_claim_makes_the_folder_and_names_this_process(tmp_path):
    out = tmp_path / "deep" / "out"
    claim = mcp_lock.claim_new(out)
    assert lock_of(out).read_text().strip() == str(os.getpid())
    claim.release()
    assert not lock_of(out).exists()
    assert (out / mcp_record.WORKSPACE_DIR).is_dir()  # the session folder stays


def test_only_one_of_many_simultaneous_claims_wins(tmp_path):
    out, wins, losses = tmp_path / "out", [], []
    start = threading.Barrier(12)

    def claim():
        start.wait()
        try:
            wins.append(mcp_lock.claim_new(out))
        except mcp_lock.InUse:
            losses.append(1)

    threads = [threading.Thread(target=claim) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert (len(wins), len(losses)) == (1, 11)
    wins[0].release()


def test_only_one_of_many_claims_of_a_leftover_moves_it(tmp_path):
    out, wins, losses = tmp_path / "out", [], []
    (out / mcp_record.WORKSPACE_DIR).mkdir(parents=True)
    mcp_turns.mark(out / mcp_record.WORKSPACE_DIR)
    (out / mcp_record.WORKSPACE_DIR / "old.txt").write_text("x")
    start = threading.Barrier(12)

    def claim():
        start.wait()
        try:
            wins.append(mcp_lock.claim_new(out, lambda: True))
        except mcp_lock.InUse:
            losses.append(1)

    threads = [threading.Thread(target=claim) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert (len(wins), len(losses)) == (1, 11)
    assert sorted(p.name for p in (out / mcp_record.WORKSPACE_DIR).iterdir()) == [
        mcp_lock.LOCK_NAME,
        mcp_turns.STATE_NAME,
    ]
    wins[0].release()


def test_a_folder_with_a_session_folder_is_in_use_for_a_new_claim(tmp_path):
    out = tmp_path / "out"
    (out / mcp_record.WORKSPACE_DIR).mkdir(parents=True)
    with pytest.raises(mcp_lock.InUse):
        mcp_lock.claim_new(out)


def test_a_live_process_holds_the_folder(tmp_path, foreign):
    out = tmp_path / "out"
    (out / mcp_record.WORKSPACE_DIR).mkdir(parents=True)
    lock_of(out).write_text(f"{foreign}\n")
    with pytest.raises(mcp_lock.InUse) as refused:
        mcp_lock.claim_existing(out)
    assert refused.value.pid == foreign
    assert lock_of(out).read_text().strip() == str(foreign)  # untouched


def test_a_lock_of_a_process_that_is_gone_is_taken_over(tmp_path):
    out = tmp_path / "out"
    (out / mcp_record.WORKSPACE_DIR).mkdir(parents=True)
    lock_of(out).write_text(f"{dead_pid()}\n")
    claim = mcp_lock.claim_existing(out)
    assert lock_of(out).read_text().strip() == str(os.getpid())
    assert [p.name for p in lock_of(out).parent.iterdir()] == [mcp_lock.LOCK_NAME]
    claim.release()


def test_a_lock_naming_this_process_that_nothing_here_holds_is_stale(tmp_path):
    out = tmp_path / "out"
    (out / mcp_record.WORKSPACE_DIR).mkdir(parents=True)
    lock_of(out).write_text(f"{os.getpid()}\n")  # left by an earlier process
    mcp_lock.claim_existing(out).release()


def test_a_lock_this_process_holds_is_not_taken_twice(tmp_path):
    out = tmp_path / "out"
    claim = mcp_lock.claim_new(out)
    with pytest.raises(mcp_lock.InUse) as refused:
        mcp_lock.claim_existing(out)
    assert refused.value.pid == os.getpid()
    claim.release()
    mcp_lock.claim_existing(out).release()


def test_an_unreadable_lock_is_judged_by_its_age(tmp_path):
    out = tmp_path / "out"
    (out / mcp_record.WORKSPACE_DIR).mkdir(parents=True)
    lock_of(out).write_text("")  # being written right now
    with pytest.raises(mcp_lock.InUse):
        mcp_lock.claim_existing(out)
    old = time.time() - 3600
    os.utime(lock_of(out), (old, old))  # a crash between create and write
    mcp_lock.claim_existing(out).release()


def test_two_takeovers_of_one_stale_lock_leave_one_holder(tmp_path):
    for round_ in range(25):  # a race: many rounds, not one lucky one
        out = tmp_path / f"out{round_}"
        (out / mcp_record.WORKSPACE_DIR).mkdir(parents=True)
        lock_of(out).write_text(f"{dead_pid()}\n")
        wins, start = [], threading.Barrier(8)

        def take():
            start.wait()
            try:
                wins.append(mcp_lock.claim_existing(out))
            except mcp_lock.InUse:
                pass

        threads = [threading.Thread(target=take) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert len(wins) == 1, round_
        wins[0].release()


def test_release_removes_only_its_own_lock_and_twice_is_fine(tmp_path, foreign):
    out = tmp_path / "out"
    claim = mcp_lock.claim_new(out)
    claim.release()
    lock_of(out).write_text(f"{foreign}\n")  # another process took the folder since
    claim.release()
    assert lock_of(out).read_text().strip() == str(foreign)


def test_a_claim_is_registered_before_its_pid_is_written(tmp_path, monkeypatch):
    # A thread that reads the pid in between must find the claim live: judged
    # stale, it would be taken over and two callers would hold the folder.
    out = tmp_path / "out"
    (out / mcp_record.WORKSPACE_DIR).mkdir(parents=True)
    seen = []
    real = os.fdopen

    def fdopen(fd, *args, **kwargs):
        seen.append(mcp_lock._key(lock_of(out)) in mcp_lock._held)
        return real(fd, *args, **kwargs)

    monkeypatch.setattr(mcp_lock.os, "fdopen", fdopen)
    mcp_lock.claim_existing(out).release()
    assert seen == [True]


def test_threads_that_claim_and_release_one_folder_never_hold_it_together(tmp_path):
    out = tmp_path / "out"
    (out / mcp_record.WORKSPACE_DIR).mkdir(parents=True)
    holding, doubled, missing = [0], [], []
    count = threading.Lock()
    stop = time.monotonic() + 3

    def churn():
        while time.monotonic() < stop:
            try:
                claim = mcp_lock.claim_existing(out)
            except mcp_lock.InUse:
                continue
            with count:
                holding[0] += 1
                if holding[0] > 1:
                    doubled.append(1)
            if not lock_of(out).exists():
                missing.append(1)
            time.sleep(0.0005)
            with count:
                holding[0] -= 1
            claim.release()

    threads = [threading.Thread(target=churn) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert (len(doubled), len(missing)) == (0, 0)


def test_a_folder_spelled_in_another_case_is_the_same_claim(tmp_path):
    out = tmp_path / "Out"
    claim = mcp_lock.claim_new(out)
    other = tmp_path / "out"
    try:
        if not other.exists():
            pytest.skip("the file system tells letter case apart")
        with pytest.raises(mcp_lock.InUse):
            mcp_lock.claim_existing(other)
        assert lock_of(out).read_text().strip() == str(os.getpid())
    finally:
        claim.release()
    assert not lock_of(out).exists()
