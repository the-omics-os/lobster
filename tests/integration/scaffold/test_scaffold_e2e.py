"""
End-to-end test: scaffold -> install -> discover.

This test generates a plugin package, installs it in the current venv,
verifies ComponentRegistry discovers it via subprocess, then uninstalls.

Requires: running in a venv with lobster-ai installed.
Mark: slow (installs/uninstalls packages)
"""

import subprocess
import sys

import pytest

from lobster.scaffold import scaffold_agent


@pytest.mark.slow
@pytest.mark.integration
class TestScaffoldE2E:

    def test_scaffold_install_discover(self, tmp_path):
        """Generated plugin is discoverable after editable install."""
        # Generate
        pkg_dir = scaffold_agent(
            name="test_scaffold_expert",
            display_name="Test Scaffold Expert",
            description="Integration test agent",
            tier="free",
            output_dir=tmp_path,
        )

        try:
            # Install
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(pkg_dir)],
                capture_output=True,
                text=True,
                check=True,
            )

            # Discover via subprocess (fresh Python process sees new entry points)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from lobster.core.component_registry import component_registry; "
                        "component_registry.reset(); "
                        "agents = component_registry.list_agents(); "
                        "assert 'test_scaffold_expert' in agents, f'Not found in {list(agents.keys())}'; "
                        "config = agents['test_scaffold_expert']; "
                        "assert config.tier_requirement == 'free'; "
                        "assert config.supervisor_accessible is True; "
                        "print('PASS: test_scaffold_expert discovered with tier=free')"
                    ),
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (
                f"Discovery failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
            assert "PASS" in result.stdout

        finally:
            # Cleanup: uninstall
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "uninstall",
                    "-y",
                    "lobster-test-scaffold",
                ],
                capture_output=True,
            )
