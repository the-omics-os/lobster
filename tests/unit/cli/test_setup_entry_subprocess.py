"""Real module-entry output contracts, independent of pytest's shared console.

Synthetic credentials only; no model/account calls or installation. Source-entry
coverage is not installed-wheel, native-terminal, or authentication acceptance.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.no_auto_config


def test_real_init_entry_reports_missing_key_and_existing_config(tmp_path):
    root = Path(__file__).resolve().parents[3]
    namespace_roots = sorted(
        path.parent for path in (root / "packages").glob("*/lobster") if path.is_dir()
    )
    for case in ("missing-key", "existing-config"):
        work = tmp_path / case
        home = work / "home"
        home.mkdir(parents=True)
        env = {
            "PATH": os.defpath,
            "HOME": str(home),
            "USERPROFILE": str(home),
            "APPDATA": str(home / "appdata"),
            "XDG_CONFIG_HOME": str(home / "config"),
            "TMPDIR": str(work),
            "TMP": str(work),
            "TEMP": str(work),
            "PYTHONPATH": os.pathsep.join(map(str, [root, *namespace_roots])),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "NUMBA_CACHE_DIR": str(work / "numba"),
            "MPLCONFIGDIR": str(work / "mpl"),
            "AWS_ACCESS_KEY_ID": "testing",
            "AWS_SECRET_ACCESS_KEY": "testing",
            "AWS_DEFAULT_REGION": "us-east-1",
            "AWS_EC2_METADATA_DISABLED": "true",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
        if os.name == "nt":
            # Required by Windows process startup, not a provider credential.
            env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
        options = ["--use-omics-os"]
        original = b"LOBSTER_LLM_PROVIDER=anthropic\nANTHROPIC_API_KEY=synthetic\n"
        if case == "existing-config":
            (work / ".env").write_bytes(original)
            options += ["--omics-os-key", "omk_subprocess_synthetic"]
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "lobster",
                "init",
                "--non-interactive",
                "--skip-extras",
                "--skip-ssl-test",
                *options,
            ],
            cwd=work,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=40,
            check=False,
        )
        (work / "stdout.txt").write_text(result.stdout)
        (work / "stderr.txt").write_text(result.stderr)
        assert result.returncode == 1, (result.stdout, result.stderr)
        assert "Traceback" not in result.stderr
        assert "setup complete" not in result.stdout.lower()
        for marker in ("Configuration saved", "API key saved"):
            assert marker.lower() not in (result.stdout + result.stderr).lower()
        assert not (work / ".lobster_workspace" / "provider_config.json").exists()
        assert not list(work.glob(".env.backup.*"))
        if case == "missing-key":
            assert "API key" in result.stdout
            assert "--omics-os-key" in result.stdout
            assert "app.omics-os.com" in result.stdout
            assert not (work / ".env").exists()
        else:
            assert "existing configuration found" in result.stdout.lower()
            assert "--force" in result.stdout
            assert (work / ".env").read_bytes() == original
