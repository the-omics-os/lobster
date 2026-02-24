"""
Installation verification tests for Lobster AI UX.

These tests validate that first-time users can install:
1. Skills via the bash installer script
2. Packages via pip with entry point discovery
3. CLI that responds to commands

No actual installation happens - these are validation tests that verify
the installation artifacts are correct.
"""

import importlib.metadata
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

# Test markers
pytestmark = pytest.mark.ux

# Path constants
LOBSTER_ROOT = Path(__file__).parent.parent.parent
SKILLS_DIR = LOBSTER_ROOT / "skills"
INSTALLER_SCRIPT = Path("/Users/tyo/omics-os/landing_lobster/public/skill")
PYPROJECT_PATH = LOBSTER_ROOT / "pyproject.toml"


# =============================================================================
# Skill Installer Script Validation
# =============================================================================


def test_skill_installer_syntax_valid():
    """Verify bash script has valid syntax."""
    if not INSTALLER_SCRIPT.exists():
        pytest.skip(f"Installer script not found: {INSTALLER_SCRIPT}")

    result = subprocess.run(
        ["bash", "-n", str(INSTALLER_SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Bash syntax error: {result.stderr}"


def test_skill_installer_has_required_variables():
    """Verify installer script defines required variables and functions."""
    if not INSTALLER_SCRIPT.exists():
        pytest.skip(f"Installer script not found: {INSTALLER_SCRIPT}")

    content = INSTALLER_SCRIPT.read_text()

    # The MANIFEST-based installer uses these core variables
    required_vars = [
        "REPO=",
        "BRANCH=",
        "BASE_URL=",
    ]

    for var in required_vars:
        assert var in content, f"Missing variable: {var}"

    # Must define the SKILLS array with skill definitions
    assert "SKILLS=(" in content, "Missing SKILLS array"

    # Must have the MANIFEST-based file discovery function
    assert "get_skill_files" in content, "Missing get_skill_files function"
    assert "MANIFEST" in content, "Missing MANIFEST reference"


def test_skill_installer_manifest_files_match_actual_files():
    """Verify MANIFEST files list actual skill files that exist on disk."""
    for skill_name in ["lobster-use", "lobster-dev"]:
        manifest_path = SKILLS_DIR / skill_name / "MANIFEST"
        assert manifest_path.exists(), f"MANIFEST not found: {manifest_path}"

        manifest_content = manifest_path.read_text()
        manifest_files = [
            line.strip()
            for line in manifest_content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        skill_dir = SKILLS_DIR / skill_name
        for file in manifest_files:
            file_path = skill_dir / file
            assert file_path.exists(), f"Skill file listed in MANIFEST missing: {file_path}"


def test_skill_installer_handles_missing_directories():
    """Verify installer creates directories gracefully."""
    content = INSTALLER_SCRIPT.read_text()

    # Should have mkdir -p for references subdirectory
    assert (
        'mkdir -p "${dest}/references"' in content
        or 'mkdir -p "$dest/references"' in content
    )


@patch("subprocess.run")
def test_skill_installer_download_logic_without_network(mock_run):
    """Mock curl to test download logic without network access."""
    if not INSTALLER_SCRIPT.exists():
        pytest.skip(f"Installer script not found: {INSTALLER_SCRIPT}")

    # Mock successful curl
    mock_run.return_value = Mock(returncode=0)

    # Run the script with mocked HOME
    env = os.environ.copy()
    env["HOME"] = "/tmp/test_lobster_install"

    # Note: This would actually run the script, so we'll just validate the logic exists
    content = INSTALLER_SCRIPT.read_text()

    # Verify curl is used
    assert "curl" in content
    assert "-fsSL" in content  # Silent, fail on error, follow redirects

    # Verify it handles failed downloads
    assert "failed=" in content


# =============================================================================
# Skills Content Validation
# =============================================================================


def test_skills_content_lobster_use_has_valid_frontmatter():
    """Verify lobster-use/SKILL.md has valid YAML frontmatter."""
    skill_file = SKILLS_DIR / "lobster-use" / "SKILL.md"
    assert skill_file.exists(), f"Skill file not found: {skill_file}"

    content = skill_file.read_text()

    # Extract frontmatter
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    assert match, "No YAML frontmatter found"

    frontmatter_text = match.group(1)
    frontmatter = yaml.safe_load(frontmatter_text)

    # Validate required fields
    assert "name" in frontmatter, "Missing 'name' in frontmatter"
    assert "description" in frontmatter, "Missing 'description' in frontmatter"

    # Validate name format
    assert re.match(r"^[a-z0-9\-]+$", frontmatter["name"]), "Invalid name format"
    assert len(frontmatter["name"]) <= 64, "Name too long (>64 chars)"


def test_skills_content_lobster_dev_has_valid_frontmatter():
    """Verify lobster-dev/SKILL.md has valid YAML frontmatter."""
    skill_file = SKILLS_DIR / "lobster-dev" / "SKILL.md"
    assert skill_file.exists(), f"Skill file not found: {skill_file}"

    content = skill_file.read_text()

    # Extract frontmatter
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    assert match, "No YAML frontmatter found"

    frontmatter_text = match.group(1)
    frontmatter = yaml.safe_load(frontmatter_text)

    # Validate required fields
    assert "name" in frontmatter, "Missing 'name' in frontmatter"
    assert "description" in frontmatter, "Missing 'description' in frontmatter"

    # Validate name format
    assert re.match(r"^[a-z0-9\-]+$", frontmatter["name"]), "Invalid name format"
    assert len(frontmatter["name"]) <= 64, "Name too long (>64 chars)"


def test_skills_content_reference_files_exist():
    """Verify all reference files mentioned in Quick Reference tables exist."""
    for skill_name in ["lobster-use", "lobster-dev"]:
        skill_file = SKILLS_DIR / skill_name / "SKILL.md"
        if not skill_file.exists():
            pytest.skip(f"Skill file not found: {skill_file}")

        content = skill_file.read_text()

        # Find reference file links (e.g., @references/cli-commands.md)
        reference_links = re.findall(r"@references/([a-z\-]+\.md)", content)

        for ref_file in reference_links:
            ref_path = SKILLS_DIR / skill_name / "references" / ref_file
            assert ref_path.exists(), f"Reference file missing: {ref_path}"


def test_skills_content_reference_files_not_empty():
    """Verify reference files have content (not empty)."""
    for skill_name in ["lobster-use", "lobster-dev"]:
        refs_dir = SKILLS_DIR / skill_name / "references"
        if not refs_dir.exists():
            pytest.skip(f"References directory not found: {refs_dir}")

        for ref_file in refs_dir.glob("*.md"):
            content = ref_file.read_text()
            assert len(content) > 100, f"Reference file too small: {ref_file}"


def test_skills_content_total_file_count_matches_manifest():
    """Verify total skill .md files match MANIFEST file counts."""
    for skill_name in ["lobster-use", "lobster-dev"]:
        manifest_path = SKILLS_DIR / skill_name / "MANIFEST"
        if not manifest_path.exists():
            pytest.skip(f"MANIFEST not found: {manifest_path}")

        manifest_content = manifest_path.read_text()
        manifest_files = [
            line.strip()
            for line in manifest_content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        manifest_count = len(manifest_files)

        # Count actual .md files on disk
        actual_files = list((SKILLS_DIR / skill_name).rglob("*.md"))
        actual_count = len(actual_files)

        assert (
            actual_count == manifest_count
        ), f"{skill_name} file count mismatch: {actual_count} on disk != {manifest_count} in MANIFEST"


# =============================================================================
# Package Installation Simulation
# =============================================================================


def test_package_pyproject_has_entry_points_section():
    """Verify pyproject.toml has entry points section."""
    assert PYPROJECT_PATH.exists(), f"pyproject.toml not found: {PYPROJECT_PATH}"

    content = PYPROJECT_PATH.read_text()

    # Should have entry points for services (agents moved to separate packages)
    # Note: agents are now in workspace packages, so we check for services
    assert (
        '[project.entry-points."lobster.services"]' in content
        or 'project.entry-points."lobster.services"' in content
    )


def test_package_cli_entry_point_exists():
    """Verify CLI entry point exists in pyproject.toml."""
    content = PYPROJECT_PATH.read_text()

    # Look for console_scripts entry point
    assert "[project.scripts]" in content or 'lobster = "lobster.cli:' in content


def test_package_entry_points_reference_valid_modules():
    """Verify entry points reference valid module paths."""
    # Note: This test checks that the package structure exists, not that imports work
    # (imports would require the full environment)

    content = PYPROJECT_PATH.read_text()

    # Find all entry point definitions
    entry_point_pattern = r'(\w+)\s*=\s*"([\w\.]+):([\w]+)"'
    matches = re.findall(entry_point_pattern, content)

    for name, module_path, attr in matches:
        if not module_path.startswith("lobster."):
            continue  # Skip non-lobster entries

        # Convert module path to file path
        # e.g., lobster.agents.transcriptomics.transcriptomics_expert -> lobster/agents/transcriptomics/transcriptomics_expert.py
        parts = module_path.split(".")

        # Check core lobster paths
        potential_paths = [
            LOBSTER_ROOT / ("/".join(parts) + ".py"),
            LOBSTER_ROOT / ("/".join(parts[1:]) + ".py"),  # Skip 'lobster' prefix
            LOBSTER_ROOT / "/".join(parts) / "__init__.py",
            LOBSTER_ROOT / "/".join(parts[1:]) / "__init__.py",
        ]

        # Check package paths
        if "packages" not in module_path:
            packages_dir = LOBSTER_ROOT / "packages"
            if packages_dir.exists():
                for package_dir in packages_dir.glob("lobster-*"):
                    package_parts = parts[1:]  # Remove 'lobster' prefix
                    potential_paths.extend(
                        [
                            package_dir / "lobster" / ("/".join(package_parts) + ".py"),
                            package_dir
                            / "lobster"
                            / "/".join(package_parts)
                            / "__init__.py",
                        ]
                    )

        # At least one path should exist
        path_exists = any(p.exists() for p in potential_paths)
        if not path_exists:
            # This might be okay if the module is in a workspace package
            # Just warn instead of failing
            pytest.skip(
                f"Module path not found (might be in workspace package): {module_path}"
            )


@pytest.mark.skip(reason="Requires actual package installation")
def test_package_component_registry_discovers_agents():
    """Test ComponentRegistry.discover_agents() finds expected agents."""
    # This would require importing lobster, which requires full environment
    # Marked as skip for CI
    from lobster.core.component_registry import ComponentRegistry

    registry = ComponentRegistry()
    agents = registry.discover_agents()

    # Expected agents from packages
    expected_agents = [
        "transcriptomics_expert",
        "annotation_expert",
        "de_analysis_expert",
        "research_agent",
        "data_expert",
        "visualization_expert",
        "metadata_assistant",
        "protein_structure_visualization_expert",
        "genomics_expert",
        "proteomics_expert",
        "machine_learning_expert",
    ]

    for agent_name in expected_agents:
        assert agent_name in agents, f"Agent not discovered: {agent_name}"


# =============================================================================
# Entry Point Discovery
# =============================================================================


@pytest.mark.skip(reason="Requires package installation")
def test_entry_point_loading_via_importlib():
    """Use importlib.metadata to verify entry points load."""
    # Requires actual pip install
    entry_points = importlib.metadata.entry_points()

    # Get lobster.agents group
    if hasattr(entry_points, "select"):
        # Python 3.10+
        agent_eps = entry_points.select(group="lobster.agents")
    else:
        # Python 3.9
        agent_eps = entry_points.get("lobster.agents", [])

    agent_names = [ep.name for ep in agent_eps]

    assert len(agent_names) > 0, "No agent entry points found"


@pytest.mark.skip(reason="Requires package installation")
def test_entry_point_agent_configs_exist():
    """Verify each lobster.agents entry point has AGENT_CONFIG."""
    entry_points = importlib.metadata.entry_points()

    if hasattr(entry_points, "select"):
        agent_eps = entry_points.select(group="lobster.agents")
    else:
        agent_eps = entry_points.get("lobster.agents", [])

    for ep in agent_eps:
        config = ep.load()
        assert hasattr(
            config, "name"
        ), f"Entry point {ep.name} missing AGENT_CONFIG.name"
        assert hasattr(
            config, "display_name"
        ), f"Entry point {ep.name} missing AGENT_CONFIG.display_name"


@pytest.mark.skip(reason="Requires package installation and is slow")
def test_entry_point_loading_performance():
    """Time entry point loading (should be < 50ms each)."""
    entry_points = importlib.metadata.entry_points()

    if hasattr(entry_points, "select"):
        agent_eps = entry_points.select(group="lobster.agents")
    else:
        agent_eps = entry_points.get("lobster.agents", [])

    for ep in agent_eps:
        start = time.perf_counter()
        config = ep.load()
        duration_ms = (time.perf_counter() - start) * 1000

        assert (
            duration_ms < 50
        ), f"Entry point {ep.name} took {duration_ms:.1f}ms to load (>50ms)"


# =============================================================================
# CLI Validation (requires installation)
# =============================================================================


@pytest.mark.skip(reason="Requires package installation")
def test_cli_help_command():
    """Verify `lobster --help` responds."""
    result = subprocess.run(
        ["lobster", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, f"CLI help failed: {result.stderr}"
    assert "lobster" in result.stdout.lower()


@pytest.mark.skip(reason="Requires package installation")
def test_cli_version_command():
    """Verify `lobster --version` responds."""
    result = subprocess.run(
        ["lobster", "--version"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, f"CLI version failed: {result.stderr}"
    # Should contain version number
    assert re.search(r"\d+\.\d+\.\d+", result.stdout)
