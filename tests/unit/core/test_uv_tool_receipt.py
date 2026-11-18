"""`lobster init` under `uv tool` rebuilds the tool from its own receipt.

The rebuild must keep how Lobster itself was installed (a wheel file stays that
wheel, not PyPI's latest) and every `--with` requirement, and must never write
an `==` pin: uv then refuses to upgrade the tool.
"""

import sys
from pathlib import Path

import pytest

from lobster.cli_internal.commands.heavy import init_commands
from lobster.core.uv_tool_env import build_tool_install_command

WHEEL = "/opt/wheels/lobster_ai-1.3.0+c09f12bb5-py3-none-any.whl"


@pytest.fixture
def receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _name: "uv")
    monkeypatch.setattr("lobster.version.__version__", "1.3.0+c09f12bb5")

    def _write(*lines):
        body = ",\n".join(f"    {line}" for line in lines)
        (tmp_path / "uv-receipt.toml").write_text(
            f"[tool]\nrequirements = [\n{body},\n]\n"
        )

    return _write


def test_a_wheel_install_stays_that_wheel_and_keeps_its_with_set(receipt):
    receipt(
        f'{{ name = "lobster-ai", extras = ["bedrock"], path = "{WHEEL}" }}',
        '{ name = "lobster-research", specifier = ">=1.3.0" }',
    )

    command = build_tool_install_command(extras=["lamin"])

    assert command == [
        "uv",
        "tool",
        "install",
        f"lobster-ai[bedrock,lamin] @ {Path(WHEEL).as_uri()}",
        "--with",
        "lobster-research>=1.3.0",
    ]


def test_a_plain_install_gets_a_floor_never_a_pin(receipt):
    receipt('{ name = "lobster-ai" }')

    command = build_tool_install_command(extras=["lamin"])

    assert command[3] == "lobster-ai[lamin]>=1.3.0"
    assert not any("==" in part for part in command)


def test_new_agent_packages_get_a_floor_at_lobsters_minor(receipt, monkeypatch):
    receipt('{ name = "lobster-ai", specifier = ">=1.3.0" }')
    monkeypatch.setattr(
        "lobster.core.component_registry.AGENT_TO_PACKAGE",
        {"transcriptomics_expert": "lobster-transcriptomics"},
    )

    command = init_commands._build_uv_tool_init_command(
        None, ["transcriptomics_expert"]
    )

    assert command[3:] == [
        "lobster-ai>=1.3.0",
        "--with",
        "lobster-transcriptomics>=1.3.0",
    ]
