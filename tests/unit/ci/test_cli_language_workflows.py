"""Regression coverage for language CI selection and catalog-only PRs."""

from __future__ import annotations

import os
import shlex
import subprocess  # nosec B404 # test runs a workflow shell step against a stub PATH
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


def _workflow(name: str) -> dict:
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text())


def _step(workflow: dict, job: str, name: str) -> dict:
    return next(step for step in workflow["jobs"][job]["steps"] if step["name"] == name)


@pytest.mark.parametrize(
    ("workflow", "job", "step"),
    [
        (
            "ci-basic.yml",
            "quality-and-tests",
            "Run language and upgrade regression tests",
        ),
        ("pr-validation-basic.yml", "fast-validation", "Run critical unit tests"),
    ],
)
def test_python_language_and_upgrade_suites_are_selected(workflow, job, step):
    command = shlex.split(_step(_workflow(workflow), job, step)["run"], comments=True)
    assert "tests/unit/i18n" in command
    assert "tests/unit/core/test_self_upgrade.py" in command
    assert "tests/unit/ci" in command


@pytest.mark.parametrize("workflow", ["ci-basic.yml", "pr-validation-basic.yml"])
def test_go_language_package_runs_with_tui_suites(workflow):
    document = _workflow(workflow)
    job = "go-tui-verify" if workflow == "ci-basic.yml" else "go-tui-validation"
    command = shlex.split(_step(document, job, "Run Go tests")["run"])
    assert command[:2] == ["go", "test"]
    assert "./internal/i18n" in command


@pytest.mark.parametrize(
    ("changed_paths", "expected"),
    [
        ("lobster/i18n/zh.json", "true"),
        ("lobster/i18n/phrases.zh.json", "true"),
        ("tests/unit/i18n/test_catalog.py", "true"),
        ("lobster-tui/internal/i18n/i18n.go", "true"),
        ("lobster/cli_internal/go_tui_launcher.py", "true"),
        ("lobster/ui/bridge/init_adapter.py", "true"),
        ("README.md", "false"),
        ("", "false"),
    ],
)
def test_pr_go_filter_handles_catalog_only_and_unrelated_changes(
    tmp_path, changed_paths, expected
):
    script = _step(
        _workflow("pr-validation-basic.yml"),
        "go-tui-validation",
        "Detect Go TUI changes",
    )["run"].replace("${{ github.event.pull_request.base.ref }}", "main")
    git = tmp_path / "git"
    git.write_text('#!/bin/sh\nprintf "%s\\n" "$TEST_CHANGED_PATHS"\n')
    git.chmod(0o755)
    output = tmp_path / "github-output"
    env = dict(
        os.environ,
        PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        TEST_CHANGED_PATHS=changed_paths,
        GITHUB_OUTPUT=str(output),
    )
    result = subprocess.run(  # nosec B603 B607 # fixed argv, PATH lookup intended
        ["bash", "-e", "-o", "pipefail", "-c", script],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text().strip() == f"has_changes={expected}"
