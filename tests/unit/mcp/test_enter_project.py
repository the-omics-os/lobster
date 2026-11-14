"""A run of this build reads its provider file and .env from the server's
working folder, so the server makes the root of the project it serves its
working folder: at start (from ``--workspace`` or a subfolder), and on a later
call once a project appears above the start folder. Real stdio, real engine
start; the provider is one whose package or key is missing, so no run reaches
a model."""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from lobster.config.workspace_config import WorkspaceProviderConfig
from lobster.core import projects
from tests.unit.mcp.test_run import ROOT

# conftest's autouse provider fixture would set LOBSTER_LLM_PROVIDER=anthropic
pytestmark = pytest.mark.no_auto_config

NOT_REACHED = {"missing_provider_package", "missing_credentials"}
GEMINI_ABSENT = pytest.mark.skipif(
    importlib.util.find_spec("langchain_google_genai") is not None,
    reason="the gemini package is installed here: a run would need a key",
)


class Server:
    """``lobster mcp --stdio`` in its own process, one call at a time."""

    def __init__(self, tmp: Path, cwd: Path, *args: str, argv0=None, env=None):
        env = {
            **(env or {}),
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp / "home"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT), str(ROOT / "packages" / "lobster-transcriptomics")]
            ),
        }
        command = [argv0] if argv0 else [sys.executable, "-m", "lobster"]
        self.err = open(tmp / "server.err", "w")
        self.proc = subprocess.Popen(  # nosec B603
            [*command, "mcp", "--stdio", *args],
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.err,
            text=True,
        )
        self.n = 0
        self.instructions = self.request(
            "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}}
        )["instructions"]

    def request(self, method, params):
        self.n += 1
        line = {"jsonrpc": "2.0", "id": self.n, "method": method, "params": params}
        self.proc.stdin.write(json.dumps(line) + "\n")
        self.proc.stdin.flush()
        for reply in self.proc.stdout:
            reply = json.loads(reply)
            if reply.get("id") == self.n:
                return reply["result"]
        raise AssertionError(f"no reply; stderr: {self.stderr()}")

    def tool(self, name, **args):
        result = self.request("tools/call", {"name": name, "arguments": args})
        text = result["content"][0]["text"]
        return text if result.get("isError") else json.loads(text)

    def run_ends(self, **args):
        run = self.tool("lobster_run", **args)
        assert isinstance(run, dict), run
        for _ in range(10):
            done = self.tool("lobster_wait", run_id=run["run_id"], timeout_s=60)
            if done["status"] in ("completed", "failed", "cancelled"):
                return run, done
        raise AssertionError(self.stderr())

    def stderr(self):
        return (Path(self.err.name)).read_text()[-2000:]

    def close(self):
        self.proc.stdin.close()
        assert self.proc.wait(timeout=60) == 0, self.stderr()
        self.err.close()


@pytest.fixture
def gemini_project(tmp_path):
    made, _ = projects.init_project(tmp_path / "study", name="Study")
    WorkspaceProviderConfig(global_provider="gemini").save(made.workspace)
    (made.root / "data" / "deeper").mkdir(parents=True)
    source = made.root / "data" / "pbmc.h5ad"
    source.write_bytes(b"")
    return made, source


def uses_the_project(server, source):
    seen = server.tool("lobster_overview")
    assert (seen["provider"], seen["provider_source"]) == ("gemini", "project")
    assert "project_settings_not_applied" not in seen
    run, done = server.run_ends(question="q", input_path=str(source), plan=False)
    assert "project_settings_not_applied" not in run
    assert done["error"]["code"] in NOT_REACHED, done["error"]


def test_workspace_from_elsewhere_runs_with_the_project_settings(
    tmp_path, gemini_project
):
    project, source = gemini_project
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    server = Server(tmp_path, elsewhere, "--workspace", str(project.root / "data"))
    assert server.instructions.startswith("Lobster project Study")
    uses_the_project(server, source)
    server.close()


def test_a_start_in_a_subfolder_runs_with_the_project_settings(
    tmp_path, gemini_project
):
    project, source = gemini_project
    server = Server(tmp_path, project.root / "data" / "deeper")
    uses_the_project(server, source)
    server.close()


def test_a_project_made_above_the_start_folder_is_entered_without_a_restart(
    tmp_path,
):
    parent = tmp_path / "study"
    start = parent / "data"
    start.mkdir(parents=True)
    source = start / "pbmc.h5ad"
    source.write_bytes(b"")
    server = Server(tmp_path, start)
    assert server.tool("lobster_overview")["project"] is None
    projects.init_project(parent, name="Later")
    WorkspaceProviderConfig(global_provider="gemini").save(
        parent / ".lobster_workspace"
    )
    uses_the_project(server, source)
    server.close()


def test_a_relative_executable_is_named_by_its_absolute_path(tmp_path, gemini_project):
    project, _ = gemini_project
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "bin").mkdir(parents=True)
    script = elsewhere / "bin" / "lobster"
    script.write_text(
        f"#!{sys.executable}\nimport sys\nfrom lobster.cli import app\nsys.exit(app())\n"
    )
    script.chmod(0o755)
    server = Server(
        tmp_path, elsewhere, "--workspace", str(project.root), argv0="./bin/lobster"
    )
    line = server.tool("lobster_overview")["commands"]["install_experts"]
    assert Path(shlex.split(line)[0]) == script.resolve()
    server.close()


def test_a_relative_path_in_a_tool_argument_is_refused_before_and_after_the_move(
    tmp_path,
):
    parent = tmp_path / "study"
    start = parent / "data"
    start.mkdir(parents=True)
    server = Server(tmp_path, start)
    args = {"question": "q", "input_path": "pbmc.h5ad", "output_dir": "runs/x"}
    before = server.tool("lobster_run", **args)
    projects.init_project(parent, name="Later")
    server.tool("lobster_overview")  # the server moves into the project here
    after = server.tool("lobster_run", **args)
    assert before == after == "input_path and output_dir must be absolute paths"
    server.close()


@GEMINI_ABSENT
@pytest.mark.parametrize("other", ["global file", "environment", "neither"])
def test_the_project_provider_wins_over_a_global_or_shell_provider(
    tmp_path, gemini_project, other
):
    """The project's provider file goes into the run's own session folder,
    which the engine reads before the global file and LOBSTER_LLM_PROVIDER."""
    project, source = gemini_project
    WorkspaceProviderConfig(global_provider="gemini", gemini_model="gemini-x").save(
        project.workspace
    )
    env = {}
    if other == "global file":
        home_config = tmp_path / "home" / ".config" / "lobster"
        home_config.mkdir(parents=True)
        (home_config / "providers.json").write_text('{"default_provider": "ollama"}')
    elif other == "environment":
        env = {"LOBSTER_LLM_PROVIDER": "anthropic"}
    server = Server(tmp_path, project.root, env=env)
    seen = server.tool("lobster_overview")
    assert (seen["provider"], seen["provider_source"]) == ("gemini", "project")
    assert (seen["model"], seen["model_source"]) == ("gemini-x", "project")
    assert "project_settings_not_applied" not in seen
    run, done = server.run_ends(question="q", input_path=str(source), plan=False)
    assert "project_settings_not_applied" not in run
    assert done["error"]["code"] == "missing_provider_package", done["error"]
    server.close()
