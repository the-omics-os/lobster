"""A killed run's download claim does not stall the next run in that folder."""

import os
import signal
import subprocess
import sys
import textwrap
import time
from unittest.mock import patch

import anndata as ad
import numpy as np
import pytest

from lobster.agents.data_expert.data_expert import data_expert
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.core.schemas.download_queue import DownloadQueueEntry, DownloadStatus
from lobster.testing.provenance_replay import factory_tools

MODULE = "lobster.agents.data_expert.data_expert"
ENTRY = "queue_GSE1_abc"

# The first run: claims the entry through the real orchestrator, then hangs in
# the download until it is killed.
FIRST_RUN = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    from lobster.core.runtime.data_manager import DataManagerV2
    from lobster.tools.download_orchestrator import DownloadOrchestrator

    workspace, started = Path(sys.argv[1]), Path(sys.argv[2])

    class Hangs:
        def supported_databases(self):
            return ["geo"]

        def download_dataset(self, entry, strategy_override=None):
            started.write_text("claimed")
            time.sleep(600)

    orchestrator = DownloadOrchestrator(DataManagerV2(workspace_path=workspace))
    orchestrator.register_service(Hangs())
    orchestrator.execute_download(sys.argv[3])
    """)


class Downloads:
    def supported_databases(self):
        return ["geo"]

    def download_dataset(self, entry, strategy_override=None):
        return ad.AnnData(X=np.ones((3, 2))), {"strategy_used": "auto"}, None


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL is POSIX-only")
def test_the_next_run_restarts_a_download_whose_run_was_killed(tmp_path):
    workspace = tmp_path / "ws"
    DataManagerV2(workspace_path=workspace).download_queue.add_entry(
        DownloadQueueEntry(entry_id=ENTRY, dataset_id="GSE1", database="geo")
    )
    started = tmp_path / "started"
    first = subprocess.Popen(
        [sys.executable, "-c", FIRST_RUN, str(workspace), str(started), ENTRY]
    )
    try:
        deadline = time.monotonic() + 120
        while not started.exists():
            assert first.poll() is None, "the first run ended before claiming"
            assert time.monotonic() < deadline, "the first run never claimed"
            time.sleep(0.2)
    finally:
        os.kill(first.pid, signal.SIGKILL)
        first.wait()

    dm = DataManagerV2(workspace_path=workspace)
    assert dm.download_queue.get_entry(ENTRY).status == DownloadStatus.IN_PROGRESS
    tools = factory_tools(data_expert, MODULE, dm)
    with patch(
        "lobster.tools.download_orchestrator.DownloadOrchestrator."
        "get_service_for_database",
        return_value=Downloads(),
    ):
        started_at = time.monotonic()
        reply = tools["execute_download_from_queue"].invoke({"entry_id": ENTRY})

    assert time.monotonic() - started_at < 60
    assert reply.lstrip().startswith(
        f"Entry '{ENTRY}' was left in progress by process {first.pid}, "
        "which is no longer running"
    ), reply
    assert "Download completed successfully" in reply
    assert dm.download_queue.get_entry(ENTRY).status == DownloadStatus.COMPLETED


def test_a_live_claim_names_its_process(tmp_path):
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.download_queue.add_entry(
        DownloadQueueEntry(
            entry_id=ENTRY,
            dataset_id="GSE1",
            database="geo",
            status=DownloadStatus.IN_PROGRESS,
            claimed_pid=os.getpid(),
            claimed_host="this-host",
        )
    )
    tools = factory_tools(data_expert, MODULE, dm)
    reply = tools["execute_download_from_queue"].invoke({"entry_id": ENTRY})
    assert f"is being downloaded by process {os.getpid()} on this-host since" in reply
