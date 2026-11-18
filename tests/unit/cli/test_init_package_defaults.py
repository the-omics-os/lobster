"""Every init UI preselects the same small set of analysis packages.

The Go TUI ticks research + transcriptomics; the classic prompt defaulted to
"all", which installed every package, experimental ones included.
"""

import re
from pathlib import Path

from lobster.cli_internal.agent_packages import (
    AVAILABLE_AGENT_PACKAGES,
    DEFAULT_AGENT_PACKAGES,
)
from lobster.cli_internal.commands.heavy import init_commands

WIZARD = (
    Path(__file__).resolve().parents[3] / "lobster-tui/internal/initwizard/wizard.go"
)


def test_the_python_default_matches_the_go_wizard():
    source = WIZARD.read_text()
    body = source[source.index("func initAgentPackages()") :]
    body = body[: body.index("\n}\n")]
    go_defaults = re.findall(
        r'Name:\s+"([^"]+)",(?:(?!Name:).)*?Default:\s+true', body, re.S
    )
    assert set(DEFAULT_AGENT_PACKAGES) == set(go_defaults)


def test_classic_init_defaults_to_the_small_set(monkeypatch, tmp_path):
    asked = []

    def _ask(*_args, default=None, **_kwargs):
        asked.append(default)
        return default

    monkeypatch.setattr(init_commands.Prompt, "ask", _ask)
    monkeypatch.setattr(init_commands, "_get_installed_agents", lambda: {})
    monkeypatch.setattr(
        "lobster.cli_internal.commands.light.agent_commands._uv_pip_install",
        lambda _pkg: (True, ""),
    )

    agents = init_commands._prompt_manual_agent_selection(tmp_path)

    expected = {
        agent
        for name, _, package_agents, _, _ in AVAILABLE_AGENT_PACKAGES
        if name in DEFAULT_AGENT_PACKAGES
        for agent in package_agents
    }
    assert asked and asked[0] != "all"
    assert set(agents) == expected
