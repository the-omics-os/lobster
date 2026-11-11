"""JSON output stays clean when a newer lobster-ai is on PyPI.

The update notice must never reach stdout ahead of JSON output. The version
cache is seeded with a newer release in a private home, and a dead proxy keeps
PyPI unreachable, so nothing here touches the network.
"""

import json
import os
import subprocess  # nosec B404 # runs this interpreter with fixed argv
import sys
import time
from pathlib import Path

import pytest

import lobster.cli

DEAD_PROXY = "http://127.0.0.1:9"


def _run(args, tmp_path):
    home = tmp_path / "home"
    cache = home / ".lobster" / ".version_check_cache.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(
        json.dumps({"latest_version": "99.0.0", "checked_at": time.time()}),
        encoding="utf-8",
    )
    # Run the same source tree this test imported, not whatever is installed.
    source_root = str(Path(lobster.cli.__file__).resolve().parents[1])
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("LOBSTER_")},
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "PYTHON_DOTENV_DISABLED": "1",
        "PYTHONPATH": source_root,
        "PYTHONWARNINGS": "ignore",
        "HTTPS_PROXY": DEAD_PROXY,
        "HTTP_PROXY": DEAD_PROXY,
    }
    return subprocess.run(  # nosec B603 # fixed argv, no shell
        [sys.executable, "-c", "from lobster.cli import app; app()", *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=180,
    )


@pytest.mark.parametrize(
    "args",
    [
        ["query", "--json", "/help"],
        ["command", "--json", "data", "--workspace", "ws"],
    ],
    ids=["query", "command"],
)
def test_json_stdout_parses_with_a_newer_release_cached(args, tmp_path):
    proc = _run(args, tmp_path)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout)["success"] is True
