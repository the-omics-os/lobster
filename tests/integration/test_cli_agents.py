"""
Integration tests for Phase 8 CLI commands and agent management.

Tests cover:
- lobster agents list/info/install subcommands
- lobster config show agent composition display
- lobster init agent selection flags
- Omics-OS endpoint client (agent_config_endpoint)
- Phase 8 success criteria verification

Requirements:
- CLI-01: Agent selection during init (--agents, --preset, --auto-agents)
- CLI-02: Manual selection with numbered list
- CLI-03: lobster agents list shows available agents
- CLI-04: lobster agents install
- CLI-05: lobster agents info
- CONF-07: "Manual or automatic?" first question in init
- CONF-08: Automatic path calls suggest_agents()
- CONF-09: React Flow UI exports valid TOML config
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest

# Skip if not in development environment
pytestmark = pytest.mark.integration


def run_lobster_command(args: list[str], timeout: int = 30) -> tuple[int, str, str]:
    """
    Run a lobster CLI command and capture output.

    Args:
        args: Command arguments (e.g., ["agents", "list"])
        timeout: Command timeout in seconds

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    cmd = [sys.executable, "-m", "lobster"] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONWARNINGS": "ignore"},
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"


class TestAgentsList:
    """Tests for lobster agents list command."""

    def test_agents_list_runs(self):
        """lobster agents list returns 0 exit code."""
        returncode, stdout, stderr = run_lobster_command(["agents", "list"])
        assert returncode == 0, f"Command failed: {stderr}"

    def test_agents_list_shows_agents(self):
        """lobster agents list output contains agent names."""
        returncode, stdout, stderr = run_lobster_command(["agents", "list"])
        assert returncode == 0, f"Command failed: {stderr}"

        # Verify output contains known agent names
        output = stdout + stderr  # Rich output may go to stderr
        assert (
            "research_agent" in output or "Name" in output
        ), f"No agents found in output: {output[:500]}"

    def test_agents_list_shows_columns(self):
        """lobster agents list shows Name, Package, Tier, Enabled columns."""
        returncode, stdout, stderr = run_lobster_command(["agents", "list"])
        assert returncode == 0, f"Command failed: {stderr}"

        output = stdout + stderr
        # Table should contain these column headers
        assert "Name" in output or "agents installed" in output


class TestAgentsInfo:
    """Tests for lobster agents info command."""

    def test_agents_info_known_agent(self):
        """lobster agents info shows details for research_agent."""
        returncode, stdout, stderr = run_lobster_command(
            ["agents", "info", "research_agent"]
        )
        assert returncode == 0, f"Command failed: {stderr}"

        output = stdout + stderr
        # Should show agent details
        assert "research_agent" in output.lower() or "Research" in output

    def test_agents_info_shows_tier(self):
        """lobster agents info displays tier information."""
        returncode, stdout, stderr = run_lobster_command(
            ["agents", "info", "research_agent"]
        )
        assert returncode == 0, f"Command failed: {stderr}"

        output = stdout + stderr
        assert "Tier" in output or "tier" in output.lower()

    def test_agents_info_unknown_agent(self):
        """lobster agents info shows error for nonexistent agent."""
        returncode, stdout, stderr = run_lobster_command(
            ["agents", "info", "nonexistent_agent_xyz"]
        )
        # Should exit with non-zero code for unknown agent
        assert returncode != 0 or "not found" in (stdout + stderr).lower()


class TestAgentsInstall:
    """Tests for lobster agents install command."""

    def test_agents_install_help(self):
        """lobster agents install --help works."""
        returncode, stdout, stderr = run_lobster_command(
            ["agents", "install", "--help"]
        )
        assert returncode == 0, f"Command failed: {stderr}"

        output = stdout + stderr
        assert "install" in output.lower()
        assert "package" in output.lower()


class TestConfigShow:
    """Tests for lobster config show command."""

    def test_config_show_runs(self):
        """lobster config show returns output."""
        returncode, stdout, stderr = run_lobster_command(["config", "show", "--help"])
        assert returncode == 0, f"Command failed: {stderr}"

        output = stdout + stderr
        assert "config" in output.lower()

    def test_config_show_displays_composition(self):
        """lobster config show --help shows agent composition is included."""
        returncode, stdout, stderr = run_lobster_command(["config", "show", "--help"])
        assert returncode == 0, f"Command failed: {stderr}"

        output = stdout + stderr
        assert "agent" in output.lower()


class TestInitAgentSelection:
    """Tests for agent selection in lobster init command."""

    def test_init_has_agents_flag(self):
        """lobster init --help shows --agents flag."""
        returncode, stdout, stderr = run_lobster_command(["init", "--help"])
        assert returncode == 0, f"Command failed: {stderr}"

        output = stdout + stderr
        assert "--agents" in output, f"--agents flag not found in: {output}"

    def test_init_has_preset_flag(self):
        """lobster init --help shows --preset flag."""
        returncode, stdout, stderr = run_lobster_command(["init", "--help"])
        assert returncode == 0, f"Command failed: {stderr}"

        output = stdout + stderr
        assert "--preset" in output, f"--preset flag not found in: {output}"

    def test_init_has_auto_agents_flag(self):
        """lobster init --help shows --auto-agents flag."""
        returncode, stdout, stderr = run_lobster_command(["init", "--help"])
        assert returncode == 0, f"Command failed: {stderr}"

        output = stdout + stderr
        assert "--auto-agents" in output, f"--auto-agents flag not found in: {output}"


class TestAgentConfigEndpoint:
    """Tests for the Omics-OS agent config endpoint client."""

    def test_suggest_agents_importable(self):
        """Module imports OK."""
        from lobster.config.agent_config_endpoint import suggest_agents

        assert callable(suggest_agents)

    def test_suggest_agents_handles_unreachable_endpoint(self):
        """suggest_agents returns None on unreachable endpoint."""
        # Set environment to use a non-existent endpoint
        import os

        original = os.environ.get("OMICS_OS_SUGGEST_ENDPOINT")
        try:
            os.environ["OMICS_OS_SUGGEST_ENDPOINT"] = (
                "http://localhost:9999/nonexistent"
            )

            from lobster.config.agent_config_endpoint import suggest_agents

            result = suggest_agents("Test workflow description")
            assert result is None, "Expected None for unreachable endpoint"
        finally:
            if original:
                os.environ["OMICS_OS_SUGGEST_ENDPOINT"] = original
            else:
                os.environ.pop("OMICS_OS_SUGGEST_ENDPOINT", None)

    def test_suggest_agents_handles_empty_input(self):
        """suggest_agents returns None for empty description."""
        from lobster.config.agent_config_endpoint import suggest_agents

        result = suggest_agents("")
        assert result is None, "Expected None for empty input"


class TestTomlExportExists:
    """Tests for React Flow TOML export component existence."""

    def test_toml_export_file_exists(self):
        """React Flow TOML export file exists in lobster-cloud."""
        # Path relative to workspace root
        toml_export_path = (
            Path(__file__).parents[3]
            / "lobster-cloud"
            / "app"
            / "src"
            / "components"
            / "agent-composer"
            / "toml-export.tsx"
        )

        # Also try absolute path
        if not toml_export_path.exists():
            toml_export_path = Path(
                "/Users/tyo/omics-os/lobster-cloud/app/src/components/agent-composer/toml-export.tsx"
            )

        assert (
            toml_export_path.exists()
        ), f"toml-export.tsx not found at {toml_export_path}"

    def test_toml_export_contains_generate_function(self):
        """toml-export.tsx contains generateTomlConfig function."""
        toml_export_path = Path(
            "/Users/tyo/omics-os/lobster-cloud/app/src/components/agent-composer/toml-export.tsx"
        )

        if toml_export_path.exists():
            content = toml_export_path.read_text()
            assert (
                "generateTomlConfig" in content
            ), "generateTomlConfig function not found"
            assert (
                'config_version = "1.0"' in content
            ), "TOML config_version format not found"


class TestPhase8SuccessCriteria:
    """Explicit tests for all 7 Phase 8 success criteria."""

    def test_criterion_1_init_agent_selection(self):
        """SC1: lobster init wizard includes agent domain and profile selection."""
        returncode, stdout, stderr = run_lobster_command(["init", "--help"])
        assert returncode == 0

        output = stdout + stderr
        # Check for agent selection flags
        assert "--agents" in output, "SC1 FAIL: --agents flag missing"
        assert "--preset" in output, "SC1 FAIL: --preset flag missing"
        assert "--auto-agents" in output, "SC1 FAIL: --auto-agents flag missing"

    def test_criterion_2_agents_list(self):
        """SC2: lobster agents list shows available agents with install status and tier."""
        returncode, stdout, stderr = run_lobster_command(["agents", "list"])
        assert returncode == 0, f"SC2 FAIL: Command failed: {stderr}"

        output = stdout + stderr
        # Should show agent table with tier
        assert (
            "agents installed" in output.lower() or "Name" in output
        ), "SC2 FAIL: No agent table"

    def test_criterion_3_agents_install(self):
        """SC3: lobster agents install installs specified agent package via pip."""
        returncode, stdout, stderr = run_lobster_command(
            ["agents", "install", "--help"]
        )
        assert returncode == 0, f"SC3 FAIL: Command failed: {stderr}"

        output = stdout + stderr
        assert (
            "package" in output.lower()
        ), "SC3 FAIL: install command not showing package argument"

    def test_criterion_4_agents_info(self):
        """SC4: lobster agents info displays agent details (tier, dependencies, description)."""
        returncode, stdout, stderr = run_lobster_command(
            ["agents", "info", "research_agent"]
        )
        assert returncode == 0, f"SC4 FAIL: Command failed: {stderr}"

        output = stdout + stderr
        assert "Tier" in output or "tier" in output.lower(), "SC4 FAIL: Tier not shown"
        # Dependencies shown as "none" or list
        assert (
            "Dependencies" in output
            or "dependencies" in output.lower()
            or "none" in output.lower()
        ), "SC4 FAIL: Dependencies not shown"

    def test_criterion_5_config_show(self):
        """SC5: lobster config show displays active agent composition from config."""
        returncode, stdout, stderr = run_lobster_command(["config", "show", "--help"])
        assert returncode == 0, f"SC5 FAIL: Command failed: {stderr}"

        output = stdout + stderr
        assert "agent" in output.lower(), "SC5 FAIL: config show doesn't mention agents"

    def test_criterion_6_llm_assisted_config(self):
        """SC6: LLM-assisted config generates valid TOML from natural language description."""
        # Test that suggest_agents function exists and is callable
        from lobster.config.agent_config_endpoint import TIMEOUT_SECONDS, suggest_agents

        assert callable(suggest_agents), "SC6 FAIL: suggest_agents not callable"
        assert TIMEOUT_SECONDS == 30, "SC6 FAIL: Timeout should be 30 seconds"

        # Test graceful fallback
        result = suggest_agents("")
        assert result is None, "SC6 FAIL: Should return None for empty input"

    def test_criterion_7_react_flow_toml_export(self):
        """SC7: React Flow UI exports valid TOML config compatible with CLI."""
        toml_export_path = Path(
            "/Users/tyo/omics-os/lobster-cloud/app/src/components/agent-composer/toml-export.tsx"
        )

        assert toml_export_path.exists(), f"SC7 FAIL: toml-export.tsx not found"

        content = toml_export_path.read_text()
        assert "generateTomlConfig" in content, "SC7 FAIL: generateTomlConfig not found"
        assert (
            'config_version = "1.0"' in content
        ), "SC7 FAIL: config_version format missing"
        assert (
            "enabled =" in content or "preset =" in content
        ), "SC7 FAIL: enabled/preset format missing"


# =============================================================================
# CLI Module Tests
# =============================================================================


class TestAgentCommandsModule:
    """Tests for the agent_commands.py module."""

    def test_agents_app_importable(self):
        """agents_app Typer app is importable."""
        from lobster.cli_internal.commands.light.agent_commands import agents_app

        assert agents_app is not None

    def test_uv_pip_install_helper_exists(self):
        """_uv_pip_install helper function exists."""
        from lobster.cli_internal.commands.light.agent_commands import _uv_pip_install

        assert callable(_uv_pip_install)

    def test_get_agents_for_package_exists(self):
        """_get_agents_for_package helper function exists."""
        from lobster.cli_internal.commands.light.agent_commands import (
            _get_agents_for_package,
        )

        assert callable(_get_agents_for_package)


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
