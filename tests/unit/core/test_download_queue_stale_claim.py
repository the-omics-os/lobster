"""A download claim left by a process that is gone is taken back at once."""

import os
import socket
import subprocess
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import anndata as ad
import numpy as np
import pytest

from lobster.core.queues.download_queue import DownloadQueue
from lobster.core.schemas.download_queue import DownloadQueueEntry, DownloadStatus
from lobster.tools.download_orchestrator import DownloadOrchestrator


def _dead_pid():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def _queue(tmp_path, *, pid=None, host=None, age=timedelta(0), status=None):
    queue = DownloadQueue(tmp_path / "download_queue.jsonl")
    entry = DownloadQueueEntry(
        entry_id="queue_GSE1_abc",
        dataset_id="GSE1",
        database="geo",
        status=status or DownloadStatus.IN_PROGRESS,
        claimed_pid=pid,
        claimed_host=host,
    )
    entry.updated_at = datetime.now() - age
    queue.add_entry(entry)
    return queue


def test_a_claim_whose_process_is_gone_is_taken_back(tmp_path):
    pid = _dead_pid()
    queue = _queue(tmp_path, pid=pid, host=socket.gethostname())

    note = queue.reclaim_stale("queue_GSE1_abc")

    assert note.startswith(
        f"Entry 'queue_GSE1_abc' was left in progress by process {pid}, "
        "which is no longer running"
    )
    assert note.endswith("starting the download again.")
    entry = queue.get_entry("queue_GSE1_abc")
    assert entry.status == DownloadStatus.FAILED
    assert f"process {pid}" in entry.error_log[-1]


def test_a_live_claim_is_kept(tmp_path):
    queue = _queue(tmp_path, pid=os.getpid(), host=socket.gethostname())
    assert queue.reclaim_stale("queue_GSE1_abc") is None
    assert queue.get_entry("queue_GSE1_abc").status == DownloadStatus.IN_PROGRESS


def test_a_claim_from_another_host_is_kept_however_old(tmp_path):
    queue = _queue(
        tmp_path, pid=_dead_pid(), host="another-host", age=timedelta(days=3)
    )
    assert queue.reclaim_stale("queue_GSE1_abc") is None
    assert queue.get_entry("queue_GSE1_abc").status == DownloadStatus.IN_PROGRESS


@pytest.mark.parametrize(
    "age,taken", [(timedelta(hours=2), True), (timedelta(minutes=5), False)]
)
def test_a_claim_without_an_owner_is_taken_back_only_when_old(tmp_path, age, taken):
    queue = _queue(tmp_path, age=age)
    note = queue.reclaim_stale("queue_GSE1_abc")
    status = queue.get_entry("queue_GSE1_abc").status
    if taken:
        assert "no process was recorded" in note
        assert status == DownloadStatus.FAILED
    else:
        assert note is None
        assert status == DownloadStatus.IN_PROGRESS


@pytest.mark.parametrize("status", [DownloadStatus.PENDING, DownloadStatus.FAILED])
def test_an_entry_that_is_not_in_progress_is_left_alone(tmp_path, status):
    queue = _queue(tmp_path, pid=_dead_pid(), host=socket.gethostname(), status=status)
    assert queue.reclaim_stale("queue_GSE1_abc") is None
    assert queue.get_entry("queue_GSE1_abc").status == status


def test_the_orchestrator_claim_records_this_process(tmp_path):
    queue = _queue(tmp_path, status=DownloadStatus.PENDING)
    seen = {}

    class Service:
        def supported_databases(self):
            return ["geo"]

        def download_dataset(self, entry, strategy_override=None):
            claimed = queue.get_entry("queue_GSE1_abc")
            seen.update(pid=claimed.claimed_pid, host=claimed.claimed_host)
            return ad.AnnData(X=np.ones((3, 2))), {}, None

    dm = MagicMock()
    dm.modalities = {}
    dm.download_queue = queue
    orchestrator = DownloadOrchestrator(dm)
    orchestrator.register_service(Service())
    orchestrator.execute_download("queue_GSE1_abc")

    assert seen == {"pid": os.getpid(), "host": socket.gethostname()}


def test_a_claim_written_by_a_live_process_meanwhile_is_kept(tmp_path, monkeypatch):
    """The stale verdict is re-checked under the lock against the same owner."""
    dead = _dead_pid()
    queue = _queue(tmp_path, pid=dead, host=socket.gethostname())
    load = queue._load_entries
    calls = SimpleNamespace(n=0)

    def load_after_a_new_claim():
        entries = load()
        calls.n += 1
        for entry in entries:
            entry.claimed_pid = os.getpid()
        return entries

    monkeypatch.setattr(queue, "_load_entries", load_after_a_new_claim)
    assert queue.reclaim_stale("queue_GSE1_abc") is None
    assert calls.n >= 1
