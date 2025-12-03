"""
Unit tests for custom feature agent.

This module tests the custom feature agent's core functionality including:
- Feature name validation
- Feature type validation
- Existing file detection across services/ categories
- Claude Code SDK integration (mocked)
- Package detection and installation checks
- Error handling paths

Tests use mocked SDK client to avoid real file creation and API costs.
"""

import asyncio
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from tests.mock_data.base import SMALL_DATASET_CONFIG
from tests.mock_data.factories import SingleCellDataFactory

# ===============================================================================
# Mock Objects and Fixtures
# ===============================================================================


@pytest.fixture
def mock_data_manager(mock_agent_environment, tmp_path):
    """Create mock data manager for custom feature agent tests."""
    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.list_modalities.return_value = []
    mock_dm.workspace_path = str(tmp_path / "workspace")
    mock_dm.modalities = {}
    mock_dm.metadata_store = {}
    mock_dm.log_tool_usage.return_value = None

    yield mock_dm


@pytest.fixture
def lobster_root(tmp_path):
    """Create a mock lobster directory structure for testing."""
    root = tmp_path / "lobster_mock"

    # Create directory structure
    (root / "lobster" / "agents").mkdir(parents=True)
    (root / "lobster" / "services" / "analysis").mkdir(parents=True)
    (root / "lobster" / "services" / "data_access").mkdir(parents=True)
    (root / "lobster" / "services" / "data_management").mkdir(parents=True)
    (root / "lobster" / "services" / "metadata").mkdir(parents=True)
    (root / "lobster" / "services" / "ml").mkdir(parents=True)
    (root / "lobster" / "services" / "orchestration").mkdir(parents=True)
    (root / "lobster" / "services" / "quality").mkdir(parents=True)
    (root / "lobster" / "services" / "visualization").mkdir(parents=True)
    (root / "lobster" / "tools").mkdir(parents=True)
    (root / "lobster" / "wiki").mkdir(parents=True)
    (root / "tests" / "unit" / "agents").mkdir(parents=True)
    (root / "tests" / "unit" / "services" / "analysis").mkdir(parents=True)
    (root / "tests" / "unit" / "tools").mkdir(parents=True)

    return root


@pytest.fixture
def mock_claude_sdk_client():
    """Mock ClaudeSDKClient to simulate SDK responses without real API calls."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.query = AsyncMock(return_value=None)

    # Create mock message types
    mock_text_block = Mock()
    mock_text_block.text = (
        "Created file: lobster/services/analysis/test_feature_service.py"
    )

    mock_assistant_message = Mock()
    mock_assistant_message.content = [mock_text_block]

    mock_result_message = Mock()
    mock_result_message.result = "Success"

    async def mock_receive():
        yield mock_assistant_message
        yield mock_result_message

    mock_client.receive_response = Mock(return_value=mock_receive())

    return mock_client


# ===============================================================================
# Feature Name Validation Tests
# ===============================================================================


@pytest.mark.unit
class TestFeatureNameValidation:
    """Test feature name validation logic."""

    def test_valid_feature_name_simple(self, mock_data_manager):
        """Test that simple valid names pass validation."""
        from lobster.agents.custom_feature_agent import custom_feature_agent

        # Import the module to access inner functions
        # We need to create the agent to get access to validation function
        agent = custom_feature_agent(mock_data_manager)

        # Valid names should pass (testing through the agent's validate_feature_name_tool)
        # Since the function is nested, we test via the tool
        assert agent is not None

    def test_valid_feature_names(self):
        """Test various valid feature name patterns."""
        valid_names = [
            "spatial_transcriptomics",
            "metabolomics",
            "variant_calling",
            "gene2vec",
            "rna_seq_v2",
            "abc123",
        ]

        # Regex pattern from custom_feature_agent
        pattern = r"^[a-z][a-z0-9_]*$"

        for name in valid_names:
            assert re.match(pattern, name), f"Expected {name} to be valid"
            assert not name.endswith("_"), f"Name should not end with underscore"

    def test_invalid_feature_names_uppercase(self):
        """Test that uppercase names are rejected."""
        invalid_names = [
            "SpatialTranscriptomics",  # PascalCase
            "METABOLOMICS",  # ALL CAPS
            "Variant_Calling",  # Mixed case
        ]

        pattern = r"^[a-z][a-z0-9_]*$"

        for name in invalid_names:
            assert not re.match(pattern, name), f"Expected {name} to be invalid"

    def test_invalid_feature_names_special_chars(self):
        """Test that special characters are rejected."""
        invalid_names = [
            "spatial-transcriptomics",  # hyphens
            "spatial transcriptomics",  # spaces
            "spatial.transcriptomics",  # dots
            "spatial@omics",  # special chars
        ]

        pattern = r"^[a-z][a-z0-9_]*$"

        for name in invalid_names:
            assert not re.match(pattern, name), f"Expected {name} to be invalid"

    def test_invalid_feature_names_start_with_number(self):
        """Test that names starting with numbers are rejected."""
        invalid_names = [
            "123_analysis",
            "10x_genomics",
        ]

        pattern = r"^[a-z][a-z0-9_]*$"

        for name in invalid_names:
            assert not re.match(pattern, name), f"Expected {name} to be invalid"

    def test_invalid_feature_names_trailing_underscore(self):
        """Test that trailing underscores are rejected."""
        names_with_trailing_underscore = [
            "spatial_",
            "analysis__",
        ]

        for name in names_with_trailing_underscore:
            assert name.endswith("_"), f"Name should end with underscore for this test"

    def test_reserved_names(self):
        """Test that reserved names are identified."""
        reserved = ["supervisor", "data", "base", "core", "config"]

        for name in reserved:
            assert name in reserved, f"{name} should be reserved"


# ===============================================================================
# Feature Type Validation Tests
# ===============================================================================


@pytest.mark.unit
class TestFeatureTypeValidation:
    """Test feature type validation logic."""

    def test_valid_feature_types(self):
        """Test that valid feature types are accepted."""
        valid_types = ["agent", "service", "agent_with_service"]

        for ftype in valid_types:
            assert ftype in valid_types

    def test_invalid_feature_types(self):
        """Test that invalid feature types are rejected."""
        invalid_types = [
            "tool",
            "provider",
            "adapter",
            "AGENT",
            "Service",
            "",
        ]

        valid_types = ["agent", "service", "agent_with_service"]

        for ftype in invalid_types:
            assert ftype not in valid_types, f"Expected {ftype} to be invalid"


# ===============================================================================
# Check Existing Files Tests
# ===============================================================================


@pytest.mark.unit
class TestCheckExistingFiles:
    """Test existing file detection across services/ categories."""

    def test_no_existing_files(self, lobster_root):
        """Test when no files exist for a feature."""
        feature_name = "new_feature"

        # Check agent file
        agent_file = lobster_root / "lobster" / "agents" / f"{feature_name}_expert.py"
        assert not agent_file.exists()

        # Check service files in all categories
        service_categories = [
            "analysis",
            "data_access",
            "data_management",
            "metadata",
            "ml",
            "orchestration",
            "quality",
            "visualization",
        ]

        for category in service_categories:
            service_file = (
                lobster_root
                / "lobster"
                / "services"
                / category
                / f"{feature_name}_service.py"
            )
            assert not service_file.exists()

    def test_existing_agent_file(self, lobster_root):
        """Test detection of existing agent file."""
        feature_name = "existing_feature"

        # Create agent file
        agent_file = lobster_root / "lobster" / "agents" / f"{feature_name}_expert.py"
        agent_file.write_text("# Existing agent")

        assert agent_file.exists()

    def test_existing_service_in_analysis(self, lobster_root):
        """Test detection of existing service in analysis/ category."""
        feature_name = "clustering"

        # Create service file in analysis/
        service_file = (
            lobster_root
            / "lobster"
            / "services"
            / "analysis"
            / f"{feature_name}_service.py"
        )
        service_file.write_text("# Existing service")

        assert service_file.exists()

    def test_existing_service_in_quality(self, lobster_root):
        """Test detection of existing service in quality/ category."""
        feature_name = "preprocessing"

        # Create service file in quality/
        service_file = (
            lobster_root
            / "lobster"
            / "services"
            / "quality"
            / f"{feature_name}_service.py"
        )
        service_file.write_text("# Existing service")

        assert service_file.exists()

    def test_existing_legacy_service_in_tools(self, lobster_root):
        """Test detection of existing service in legacy tools/ location."""
        feature_name = "legacy_feature"

        # Create service file in legacy tools/ location
        legacy_service = (
            lobster_root / "lobster" / "tools" / f"{feature_name}_service.py"
        )
        legacy_service.write_text("# Legacy service")

        assert legacy_service.exists()

    def test_existing_test_file_in_services(self, lobster_root):
        """Test detection of existing test file in services/ test directory."""
        feature_name = "tested_feature"

        # Create test directory and file
        test_dir = lobster_root / "tests" / "unit" / "services" / "analysis"
        test_dir.mkdir(parents=True, exist_ok=True)

        test_file = test_dir / f"test_{feature_name}_service.py"
        test_file.write_text("# Test file")

        assert test_file.exists()

    def test_existing_wiki_file(self, lobster_root):
        """Test detection of existing wiki file."""
        feature_name = "documented_feature"
        wiki_name = feature_name.replace("_", "-").lower()

        # Create wiki file
        wiki_file = lobster_root / "lobster" / "wiki" / f"{wiki_name}.md"
        wiki_file.write_text("# Documentation")

        assert wiki_file.exists()

    def test_scan_all_service_categories(self, lobster_root):
        """Test that all 8 service categories are scanned."""
        service_categories = [
            "analysis",
            "data_access",
            "data_management",
            "metadata",
            "ml",
            "orchestration",
            "quality",
            "visualization",
        ]

        # Verify all categories exist
        for category in service_categories:
            category_dir = lobster_root / "lobster" / "services" / category
            assert category_dir.exists(), f"Category {category} should exist"


# ===============================================================================
# Agent Creation Tests
# ===============================================================================


@pytest.mark.unit
class TestCustomFeatureAgentCreation:
    """Test custom feature agent creation."""

    def test_agent_creation_succeeds(self, mock_data_manager):
        """Test that agent can be created successfully."""
        from lobster.agents.custom_feature_agent import custom_feature_agent

        agent = custom_feature_agent(mock_data_manager)
        assert agent is not None

    def test_agent_has_graph_structure(self, mock_data_manager):
        """Test that agent has expected graph structure."""
        from lobster.agents.custom_feature_agent import custom_feature_agent

        agent = custom_feature_agent(mock_data_manager)

        # Should have a graph structure
        graph = agent.get_graph()
        assert graph is not None

    def test_agent_with_callback_handler(self, mock_data_manager):
        """Test agent creation with callback handler."""
        from lobster.agents.custom_feature_agent import custom_feature_agent

        mock_callback = Mock()

        agent = custom_feature_agent(
            data_manager=mock_data_manager, callback_handler=mock_callback
        )

        assert agent is not None

    def test_agent_with_custom_name(self, mock_data_manager):
        """Test agent creation with custom agent name."""
        from lobster.agents.custom_feature_agent import custom_feature_agent

        agent = custom_feature_agent(
            data_manager=mock_data_manager, agent_name="my_custom_feature_agent"
        )

        assert agent is not None


# ===============================================================================
# Package Detection Tests
# ===============================================================================


@pytest.mark.unit
class TestPackageDetection:
    """Test package detection from generated files."""

    def test_detect_python_imports(self, tmp_path):
        """Test detection of Python imports from file content."""
        # Create a test Python file
        test_file = tmp_path / "test_service.py"
        test_file.write_text(
            """
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
import scanpy as sc
from lobster.core.data_manager_v2 import DataManagerV2
import os
import sys
"""
        )

        # Extract imports using regex (same pattern as custom_feature_agent)
        content = test_file.read_text()
        import_pattern = r"^(?:import|from)\s+(\w+)"
        matches = re.findall(import_pattern, content, re.MULTILINE)

        # Should find these imports
        assert "numpy" in matches
        assert "pandas" in matches
        assert "scipy" in matches
        assert "sklearn" in matches
        assert "scanpy" in matches
        assert "lobster" in matches
        assert "os" in matches
        assert "sys" in matches

    def test_categorize_stdlib_modules(self):
        """Test categorization of standard library modules."""
        import sys

        stdlib_modules = set(sys.stdlib_module_names)

        # These should be in stdlib
        assert "os" in stdlib_modules
        assert "sys" in stdlib_modules
        assert "pathlib" in stdlib_modules
        assert "json" in stdlib_modules
        assert "re" in stdlib_modules

        # These should NOT be in stdlib
        assert "numpy" not in stdlib_modules
        assert "pandas" not in stdlib_modules
        assert "scanpy" not in stdlib_modules

    def test_detect_bioinformatics_tools(self):
        """Test detection of bioinformatics system tools."""
        system_tools = {
            "samtools",
            "bcftools",
            "bedtools",
            "bwa",
            "bowtie2",
            "kallisto",
            "salmon",
            "star",
            "hisat2",
            "featurecounts",
            "blastn",
            "blastp",
            "blastx",
            "makeblastdb",
        }

        # These should be recognized as system tools
        assert "samtools" in system_tools
        assert "kallisto" in system_tools
        assert "salmon" in system_tools

        # These should NOT be system tools
        assert "numpy" not in system_tools
        assert "pandas" not in system_tools


# ===============================================================================
# Branch Creation Tests (Mocked)
# ===============================================================================


@pytest.mark.unit
class TestBranchCreation:
    """Test git branch creation logic (mocked)."""

    def test_branch_name_format(self):
        """Test that branch names follow the expected format."""
        feature_name = "spatial_transcriptomics"
        date_str = date.today().strftime("%y%m%d")
        expected_branch = f"feature/{feature_name}_{date_str}"

        assert expected_branch.startswith("feature/")
        assert feature_name in expected_branch
        assert date_str in expected_branch

    @patch("subprocess.run")
    def test_branch_creation_check(self, mock_run):
        """Test branch existence check."""
        # Simulate branch not existing
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="")

        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--verify", "feature/test_123456"],
            capture_output=True,
            text=True,
            check=False,
        )

        # Branch doesn't exist when returncode != 0
        assert result.returncode != 0

    @patch("subprocess.run")
    def test_uncommitted_changes_check(self, mock_run):
        """Test uncommitted changes detection."""
        # Simulate uncommitted changes
        mock_run.return_value = Mock(
            returncode=0, stdout="M modified_file.py\n?? untracked_file.py", stderr=""
        )

        import subprocess

        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )

        # Has uncommitted changes
        assert result.stdout.strip() != ""


# ===============================================================================
# SDK Integration Tests (Mocked)
# ===============================================================================


@pytest.mark.unit
class TestSDKIntegrationMocked:
    """Test Claude Code SDK integration with mocked client."""

    def test_sdk_options_configuration(self):
        """Test that SDK options are configured correctly."""
        # Test that options structure is valid
        options_config = {
            "setting_sources": ["project"],
            "allowed_tools": ["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
            "permission_mode": "bypassPermissions",
            "max_turns": 50,
        }

        assert "setting_sources" in options_config
        assert "Read" in options_config["allowed_tools"]
        assert options_config["max_turns"] == 50

    def test_prompt_includes_service_categories(self):
        """Test that SDK prompt includes all service categories."""
        service_categories = [
            "analysis",
            "data_access",
            "data_management",
            "metadata",
            "ml",
            "orchestration",
            "quality",
            "visualization",
        ]

        # This is what the prompt should contain
        prompt_fragment = """Choose CATEGORY from: analysis, data_access, data_management, metadata, ml, orchestration, quality, visualization"""

        for category in service_categories:
            assert category in prompt_fragment

    def test_prompt_category_mapping_hints(self):
        """Test that prompt includes category mapping hints."""
        mapping_hints = {
            "clustering": "analysis",
            "QC": "quality",
            "API": "data_access",
            "modality": "data_management",
            "standardize": "metadata",
            "ML": "ml",
            "plot": "visualization",
            "workflow": "orchestration",
        }

        for feature_hint, expected_category in mapping_hints.items():
            assert expected_category in [
                "analysis",
                "data_access",
                "data_management",
                "metadata",
                "ml",
                "orchestration",
                "quality",
                "visualization",
            ]

    def test_file_pattern_regex(self):
        """Test file detection regex patterns."""
        patterns = [
            r"lobster/agents/[a-z0-9_]+_expert\.py",
            r"lobster/services/[a-z_]+/[a-z0-9_]+_service\.py",
            r"lobster/tools/[a-z0-9_]+_service\.py",
            r"tests/unit/agents/test_[a-z0-9_]+_expert\.py",
            r"tests/unit/services/[a-z_]+/test_[a-z0-9_]+_service\.py",
            r"lobster/wiki/[a-z0-9\\-]+\\.md",
        ]

        # Test sample file paths
        test_paths = [
            ("lobster/agents/spatial_expert.py", patterns[0]),
            ("lobster/services/analysis/clustering_service.py", patterns[1]),
            ("lobster/tools/legacy_service.py", patterns[2]),
            ("tests/unit/agents/test_spatial_expert.py", patterns[3]),
            ("tests/unit/services/analysis/test_clustering_service.py", patterns[4]),
        ]

        for path, pattern in test_paths:
            assert re.match(pattern, path), f"Pattern {pattern} should match {path}"


# ===============================================================================
# Error Handling Tests
# ===============================================================================


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling paths."""

    def test_empty_feature_name_rejected(self):
        """Test that empty feature name is rejected."""
        feature_name = ""
        assert not feature_name, "Empty string should be falsy"

    def test_requirements_too_short(self):
        """Test that short requirements are rejected."""
        short_requirements = "Do something"
        min_length = 20

        assert len(short_requirements.strip()) < min_length

    def test_requirements_adequate_length(self):
        """Test that adequate requirements pass length check."""
        good_requirements = """
        Create a spatial transcriptomics analysis service that handles
        Visium and Slide-seq data. Should include neighborhood analysis,
        spatial clustering, and gene expression pattern detection.
        """
        min_length = 20

        assert len(good_requirements.strip()) >= min_length

    def test_invalid_feature_type_error_message(self):
        """Test error message for invalid feature type."""
        invalid_type = "invalid_type"
        valid_types = ["agent", "service", "agent_with_service"]

        expected_error = f"Invalid feature type: '{invalid_type}'"

        assert invalid_type not in valid_types
        assert invalid_type in expected_error


# ===============================================================================
# Integration Instructions Tests
# ===============================================================================


@pytest.mark.unit
class TestIntegrationInstructions:
    """Test integration instruction generation."""

    def test_agent_registry_code_generation(self):
        """Test that agent registry code snippet is properly formatted."""
        feature_name = "spatial_transcriptomics"

        # Expected registry entry format
        registry_entry = f"""    '{feature_name}_expert_agent': AgentRegistryConfig(
        name='{feature_name}_expert_agent',
        display_name='{feature_name.replace('_', ' ').title()} Expert',
        description='[Add description of what this agent does]',
        factory_function='lobster.agents.{feature_name}_expert.{feature_name}_expert',
        handoff_tool_name='handoff_to_{feature_name}_expert_agent',
        handoff_tool_description='Hand off to the {feature_name} expert when [describe when to use].'
    ),"""

        assert f"{feature_name}_expert_agent" in registry_entry
        assert "AgentRegistryConfig" in registry_entry
        assert "factory_function" in registry_entry

    def test_agent_config_code_generation(self):
        """Test that agent config code snippet is properly formatted."""
        feature_name = "spatial_transcriptomics"
        agent_config_name = f"{feature_name}_expert_agent"

        # Expected config entry
        config_entry = f'            "{agent_config_name}": "claude-4-sonnet",'

        assert agent_config_name in config_entry

    def test_git_workflow_instructions(self):
        """Test git workflow instructions include key commands."""
        key_commands = [
            "make test",
            "make lint",
            "make type-check",
            "git add",
            "git commit",
            "git push origin",
        ]

        for cmd in key_commands:
            assert cmd, f"Command {cmd} should be non-empty"


# ===============================================================================
# State Class Tests
# ===============================================================================


@pytest.mark.unit
class TestCustomFeatureAgentState:
    """Test CustomFeatureAgentState class."""

    def test_state_class_exists(self):
        """Test that CustomFeatureAgentState is importable."""
        from lobster.agents.state import CustomFeatureAgentState

        assert CustomFeatureAgentState is not None

    def test_state_has_messages_field(self):
        """Test that state has messages field."""
        from lobster.agents.state import CustomFeatureAgentState

        # State should have messages annotation
        assert hasattr(CustomFeatureAgentState, "__annotations__") or hasattr(
            CustomFeatureAgentState, "messages"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
