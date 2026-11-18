"""With no analysis package installed, the supervisor tells the user how to add one.

A plain `lobster-ai` install has no expert to hand work to; the reply used to
be whatever the model made up.
"""

from unittest.mock import MagicMock

from lobster.agents.supervisor import create_supervisor_prompt


def _prompt(active_agents):
    dm = MagicMock()
    dm.list_modalities.return_value = []
    dm.workspace_path = "/tmp/does-not-exist"  # nosec B108 # never created
    return create_supervisor_prompt(dm, active_agents=active_agents)


def test_no_analysis_package_names_lobster_init():
    prompt = _prompt([])

    assert "No analysis packages are installed" in prompt
    assert "`lobster init`" in prompt
    assert "`lobster agents install <package>`" in prompt


def test_installed_packages_get_no_such_line():
    assert "No analysis packages are installed" not in _prompt(["research_agent"])
