"""A non-interactive run must not build a terminal prompt it never shows."""

import os
import subprocess  # nosec B404 # runs this interpreter with fixed argv
import sys
from pathlib import Path

import lobster.cli


def test_version_with_stdin_not_a_terminal_writes_nothing_to_stderr(tmp_path):
    # Run the same source tree this test imported, not whatever is installed.
    source_root = str(Path(lobster.cli.__file__).resolve().parents[1])
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "PYTHON_DOTENV_DISABLED": "1",
        "PYTHONPATH": source_root,
        "PYTHONWARNINGS": "ignore",  # import-time deprecations are not under test
    }
    result = subprocess.run(  # nosec B603 # fixed argv, no shell
        [sys.executable, "-c", "from lobster.cli import app; app()", "--version"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "lobster version" in result.stdout
    assert result.stderr == ""
