"""An upgrade moves the agent packages with Lobster, and status shows a mix.

After `pip install -U lobster-ai`, lobster-transcriptomics stayed at 1.2.6
under a 1.3 core, and nothing said so.
"""

import sys

from lobster.core import self_upgrade


def _venv(monkeypatch, uv):
    monkeypatch.setattr("lobster.core.uv_tool_env.detect_uv_tool_env", lambda: None)
    monkeypatch.setattr(self_upgrade, "_is_virtualenv", lambda: True)
    monkeypatch.setattr(self_upgrade.shutil, "which", lambda _name: uv)
    monkeypatch.setattr(
        self_upgrade,
        "installed_agent_packages",
        lambda: {"lobster-transcriptomics": "0.0.1"},
    )


def test_uv_venv_upgrade_takes_the_agent_packages_along(monkeypatch):
    _venv(monkeypatch, "/bin/uv")

    plan = self_upgrade.detect_upgrade_plan()

    assert plan.command == [
        "/bin/uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--upgrade",
        "lobster-ai",
        "lobster-transcriptomics",
    ]


def test_pip_venv_upgrade_takes_the_agent_packages_along(monkeypatch):
    _venv(monkeypatch, None)

    plan = self_upgrade.detect_upgrade_plan()

    assert plan.command[-2:] == ["lobster-ai", "lobster-transcriptomics"]


def test_status_flags_an_agent_package_from_another_minor(monkeypatch):
    from lobster.cli_internal.commands.heavy.display_helpers import (
        build_status_blocks,
    )

    monkeypatch.setattr(
        self_upgrade,
        "installed_agent_packages",
        lambda: {"lobster-transcriptomics": "0.0.1"},
    )

    blocks = build_status_blocks()

    text = repr([b.data for b in blocks])
    assert "lobster-transcriptomics" in text and "0.0.1" in text
    assert "lobster upgrade" in text
