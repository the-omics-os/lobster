"""A real server records every run of a project in the project's catalog.

``lobster mcp --stdio`` on the real engine and a scripted model, started in a
project whose catalog was switched on the way ``lobster init --lamin`` does it
(a flag in the project's workspace ``.env``, none in the shell). Three runs in one
server process; the catalog holds all three. Needs the ``lamin`` extra.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lobster.core import projects
from tests.unit.core.catalog.test_lamin_project_runs import READBACK, _run
from tests.unit.mcp.replay_input import make_input
from tests.unit.mcp.test_continue_process import ENTRY, Host
from tests.unit.mcp.test_run import ROOT

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        importlib.util.find_spec("lamindb_setup") is None
        or importlib.util.find_spec("lamindb") is None,
        reason="needs the lamin extra (lamindb-core)",
    ),
]

DATA = Path(__file__).parent / "data"
NAMES = ["first", "second", "third"]


class LaminHost(Host):
    """``Host`` started in an existing folder, with the research agents on the path."""

    def __init__(self, script: Path, cwd: Path):
        env = dict(os.environ)
        env.update(
            LOBSTER_TEST_REPLAY_FIXTURE=str(script),
            LOBSTER_LLM_PROVIDER="ollama-cloud",
            OLLAMA_API_KEY="replay-dummy",
            PYTHONDONTWRITEBYTECODE="1",
        )
        env.pop("LOBSTER_LAMIN", None)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT / "packages" / "lobster-research"), env.get("PYTHONPATH", "")]
        )
        self.proc = subprocess.Popen(  # nosec B603
            [sys.executable, "-c", ENTRY],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
            cwd=cwd,
        )
        self.n = 0


def replay(tmp_path: Path, data: Path) -> Path:
    """The QC recipe of ``continue_first_process.json``, once per run."""
    raw = json.loads((DATA / "continue_first_process.json").read_text())
    recipe = [r for r in raw["responses"] if r["content"] != "Second turn answer."]
    raw["responses"] = recipe * len(NAMES)
    text = json.dumps(raw).replace("__INPUT__", str(data))
    path = tmp_path / "replay.json"
    path.write_text(text)
    return path


def test_three_runs_in_one_server_are_all_recorded(tmp_path):
    data = tmp_path / "input.h5ad"
    make_input(data)
    (tmp_path / "study").mkdir()
    host = LaminHost(replay(tmp_path, data), tmp_path / "study")
    project, _ = projects.init_project(tmp_path / "study", name="Study")
    (project.workspace / ".env").write_text("LOBSTER_LAMIN=1\n")
    try:
        for name in NAMES:
            run = host.call(
                "lobster_run",
                question="QC this file.",
                input_path=str(data),
                output_dir=str(project.root / "runs" / name),
            )
            assert isinstance(run, dict), run
            done = host.finished(run["run_id"])
            assert done["status"] == "completed", done
    finally:
        assert host.close() == 0
    assert list(project.root.glob("runs/*/.lobster_workspace/.lamindb")) == []
    back = _run(
        READBACK,
        tmp_path,
        env_extra={"PROJ": str(project.root)},
        drop_enable=True,
    )
    assert len(back["runs"]) == 3, back
    assert back["statuses"] == ["completed"]
    for name in NAMES:
        saved = [k for k in back["keys"] if k.startswith(f"runs/{name}/")]
        assert saved, (name, back["keys"])
    assert len(set(back["keys"])) == len(back["keys"])
    assert set(back["in_place"]) == {True}
    assert back["integrity"] == "ok"
