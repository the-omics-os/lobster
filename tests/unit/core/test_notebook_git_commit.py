"""The exported notebook's ``git_commit`` names the installed Lobster, never the
current directory's repository.

The value comes from the installed lobster-ai's PEP 610 ``direct_url.json``
(``vcs_info.commit_id``, written by a git install). Anything else, such as a
PyPI wheel, an editable or folder install, or an unreadable or malformed file,
is "unavailable".
"""

import json
import subprocess
import tempfile
from importlib import metadata
from pathlib import Path
from unittest.mock import Mock

import nbformat
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.notebooks.exporter import NotebookExporter
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.core.provenance.provenance import ProvenanceTracker

COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def _no_host_dependency_scan(monkeypatch):
    # Host package metadata is an external boundary; everything else is real.
    monkeypatch.setattr(NotebookExporter, "_snapshot_dependencies", lambda self: {})


def _exporter() -> NotebookExporter:
    dm = Mock(spec=DataManagerV2)
    dm.modalities = {}
    workspace = Path(tempfile.mkdtemp())
    dm.workspace_dir = workspace
    dm.workspace_path = workspace
    tracker = ProvenanceTracker()
    tracker.create_activity(
        activity_type="step",
        agent="test",
        ir=AnalysisStep(
            operation="test.step",
            tool_name="test_step",
            description="test step",
            library="scanpy",
            code_template="x = 1",
            imports=[],
            parameters={},
            parameter_schema={},
            exportable=True,
        ),
    )
    return NotebookExporter(tracker, dm)


def _record(nb) -> dict:
    cell = next(c for c in nb.cells if "data_integrity_manifest" in c.source)
    record = json.loads(cell.source.split("```json\n", 1)[1].split("\n```", 1)[0])
    return record["data_integrity_manifest"]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _installed(monkeypatch, tmp_path, direct_url):
    """Make metadata.distribution("lobster-ai") a dist-info whose
    direct_url.json holds ``direct_url`` (bytes as-is; None = no file)."""
    info = tmp_path / "lobster_ai-9.9.9.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: lobster-ai\nVersion: 9.9.9\n")
    if direct_url is not None:
        (info / "direct_url.json").write_bytes(direct_url)
    real = metadata.distribution

    def distribution(name):
        if name == "lobster-ai":
            return metadata.PathDistribution(info)
        return real(name)

    monkeypatch.setattr(metadata, "distribution", distribution)


def _vcs(commit_id) -> bytes:
    return json.dumps(
        {
            "url": "https://github.com/the-omics-os/lobster.git",
            "vcs_info": {"vcs": "git", "commit_id": commit_id},
        }
    ).encode()


def test_the_current_directorys_repository_is_not_named_as_lobsters_commit(
    monkeypatch, tmp_path
):
    """A user running Lobster inside their own git project: their commit is not Lobster's."""
    repo = tmp_path / "their-project"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("a")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "theirs")
    theirs = _git(repo, "rev-parse", "HEAD")
    monkeypatch.chdir(repo)

    path = _exporter().export("nb")
    with open(path) as handle:
        system = _record(nbformat.read(handle, as_version=4))["system"]

    assert system["git_commit"] != theirs[:8], system
    assert theirs[:8] not in json.dumps(system)


def test_a_git_install_names_its_commit(monkeypatch, tmp_path):
    _installed(monkeypatch, tmp_path, _vcs(COMMIT))

    assert _exporter()._get_system_info()["git_commit"] == COMMIT[:8]


@pytest.mark.parametrize(
    "direct_url",
    [
        pytest.param(None, id="no direct_url.json (PyPI wheel)"),
        pytest.param(
            json.dumps(
                {"url": "file:///src/lobster", "dir_info": {"editable": True}}
            ).encode(),
            id="editable install",
        ),
        pytest.param(b"{not json", id="malformed"),
        pytest.param(b"\xff\xfe", id="not text"),
        pytest.param(b"[]", id="not an object"),
        pytest.param(
            json.dumps({"vcs_info": []}).encode(), id="vcs_info not an object"
        ),
        pytest.param(_vcs("g" * 40), id="non-hex"),
        pytest.param(_vcs(COMMIT.upper()), id="upper-case hex"),
        pytest.param(_vcs(COMMIT[:39]), id="39 characters"),
        pytest.param(_vcs(COMMIT + "0"), id="41 characters"),
        pytest.param(_vcs(COMMIT + "\n"), id="trailing newline"),
        pytest.param(_vcs(int("1" * 40)), id="not a string"),
        pytest.param(_vcs(None), id="null"),
    ],
)
def test_anything_but_a_git_commit_is_unavailable(monkeypatch, tmp_path, direct_url):
    _installed(monkeypatch, tmp_path, direct_url)

    assert _exporter()._get_system_info()["git_commit"] == "unavailable"


def test_an_uninstalled_lobster_is_unavailable(monkeypatch):
    def distribution(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", distribution)

    assert _exporter()._get_system_info()["git_commit"] == "unavailable"


def test_a_failing_metadata_read_does_not_fail_the_export(monkeypatch):
    def distribution(name):
        raise RuntimeError("unreadable site-packages")

    monkeypatch.setattr(metadata, "distribution", distribution)

    path = _exporter().export("nb")
    with open(path) as handle:
        system = _record(nbformat.read(handle, as_version=4))["system"]

    assert system["git_commit"] == "unavailable"
