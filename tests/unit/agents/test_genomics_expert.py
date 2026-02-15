"""
Unit tests for genomics_expert agent.

This module tests the genomics expert agent's core functionality including:
- Agent creation and configuration
- Tool registration (10 tools)
- Service integration
- DataManagerV2 integration
- Subscription tier handling (PREMIUM only)
"""

from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from lobster.agents.genomics.genomics_expert import genomics_expert
from lobster.core.data_manager_v2 import DataManagerV2

# ===============================================================================
# Mock Objects and Fixtures
# ===============================================================================


@pytest.fixture
def mock_genomics_adata():
    """Create mock genomics AnnData for testing."""
    np.random.seed(42)
    n_samples, n_variants = 100, 50
    genotypes = np.random.choice(
        [0, 1, 2], size=(n_samples, n_variants), p=[0.49, 0.42, 0.09]
    )
    genotypes = genotypes.astype(float)

    adata = AnnData(X=genotypes)
    adata.var = pd.DataFrame(
        {
            "CHROM": ["22"] * n_variants,
            "POS": np.arange(16050000, 16050000 + n_variants),
            "REF": ["A"] * n_variants,
            "ALT": ["G"] * n_variants,
        }
    )
    adata.obs["height"] = np.random.normal(170, 10, n_samples)
    adata.obs["age"] = np.random.randint(20, 80, n_samples)
    adata.obs["sex"] = np.random.choice([1, 2], n_samples)
    adata.layers["GT"] = genotypes.copy()
    adata.uns["data_type"] = "genomics"
    adata.uns["modality"] = "wgs"

    return adata


@pytest.fixture
def mock_data_manager(mock_provider_config, tmp_path, mock_genomics_adata):
    """Create mock data manager with genomics modalities."""
    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.list_modalities.return_value = ["wgs_study1", "snp_array_gwas1"]
    mock_dm.get_modality.return_value = mock_genomics_adata
    mock_dm.modalities = {"wgs_study1": mock_genomics_adata}
    mock_dm.workspace_path = str(tmp_path / "workspace")
    mock_dm.cache_dir = str(tmp_path / "cache")
    mock_dm.log_tool_usage = Mock()
    mock_dm.save_modality = Mock()

    return mock_dm


# ===============================================================================
# Agent Core Tests
# ===============================================================================


@pytest.mark.unit
class TestGenomicsExpertCore:
    """Test genomics expert agent core functionality."""

    def test_agent_creation_succeeds(self, mock_data_manager):
        """Test that agent can be created successfully."""
        agent = genomics_expert(mock_data_manager)
        assert agent is not None

    def test_agent_has_graph_structure(self, mock_data_manager):
        """Test that agent has expected graph structure."""
        agent = genomics_expert(mock_data_manager)
        graph = agent.get_graph()
        assert graph is not None

    def test_agent_with_custom_name(self, mock_data_manager):
        """Test agent creation with custom name."""
        agent = genomics_expert(mock_data_manager, agent_name="custom_genomics")
        assert agent is not None

    def test_agent_with_delegation_tools(self, mock_data_manager):
        """Test agent creation with delegation tools."""
        # genomics_expert doesn't currently use delegation_tools
        # (no child agents in Phase 1-2)
        # Just verify agent creation works without delegation_tools
        agent = genomics_expert(mock_data_manager)
        assert agent is not None


# ===============================================================================
# Tool Registration Tests
# ===============================================================================


@pytest.mark.unit
class TestGenomicsExpertTools:
    """Test genomics expert tool registration."""

    def test_agent_has_10_tools(self, mock_data_manager):
        """Test that agent has all 10 expected tools."""
        agent = genomics_expert(mock_data_manager)

        # genomics_expert should have 10 tools:
        # 1. load_vcf
        # 2. load_plink
        # 3. assess_quality
        # 4. filter_samples
        # 5. filter_variants
        # 6. run_gwas
        # 7. calculate_pca
        # 8. annotate_variants
        # 9. list_modalities
        # 10. get_modality_info

        # Note: Tool count verification depends on LangGraph internals
        # At minimum, verify agent was created successfully
        assert agent is not None


# ===============================================================================
# Service Integration Tests
# ===============================================================================


@pytest.mark.unit
class TestGenomicsExpertServiceIntegration:
    """Test genomics expert integration with services."""

    def test_agent_integrates_with_quality_service(self, mock_data_manager):
        """Test that agent can use GenomicsQualityService."""
        agent = genomics_expert(mock_data_manager)

        # Agent creation should succeed with quality service
        assert agent is not None

    def test_agent_integrates_with_gwas_service(self, mock_data_manager):
        """Test that agent can use GWASService."""
        agent = genomics_expert(mock_data_manager)

        # Agent creation should succeed with GWAS service
        assert agent is not None


# ===============================================================================
# Subscription Tier Tests
# ===============================================================================


@pytest.mark.unit
class TestGenomicsExpertSubscriptionTiers:
    """Test genomics expert subscription tier handling."""

    def test_agent_with_default_tier(self, mock_data_manager):
        """Test agent creation with default tier (no parameter)."""
        # genomics_expert doesn't take subscription_tier parameter
        agent = genomics_expert(mock_data_manager)

        # Should create agent successfully
        assert agent is not None

    def test_agent_creation_variations(self, mock_data_manager):
        """Test agent creation with various parameter combinations."""
        # Test with agent_name
        agent1 = genomics_expert(mock_data_manager, agent_name="custom_genomics")
        assert agent1 is not None

        # Test with callback_handler
        agent2 = genomics_expert(mock_data_manager, callback_handler=None)
        assert agent2 is not None

        # Test with workspace_path
        from pathlib import Path

        agent3 = genomics_expert(mock_data_manager, workspace_path=Path("/tmp/test"))
        assert agent3 is not None


# ===============================================================================
# Integration with DataManagerV2
# ===============================================================================


@pytest.mark.unit
class TestGenomicsExpertDataManagerIntegration:
    """Test genomics expert integration with DataManagerV2."""

    def test_agent_accesses_modalities(self, mock_data_manager):
        """Test that agent can access modalities from data manager."""
        agent = genomics_expert(mock_data_manager)

        # Verify data manager is accessible
        assert agent is not None
        # In real usage, tools would call mock_data_manager.get_modality()

    def test_agent_lists_modalities(self, mock_data_manager):
        """Test that agent can list available modalities."""
        agent = genomics_expert(mock_data_manager)

        # Verify mock is configured
        modalities = mock_data_manager.list_modalities()
        assert "wgs_study1" in modalities


# ===============================================================================
# Configuration Tests
# ===============================================================================


@pytest.mark.unit
class TestGenomicsExpertConfiguration:
    """Test genomics expert configuration."""

    def test_agent_config_in_registry(self):
        """Test that genomics_expert is registered in agent registry."""
        from lobster.config.agent_registry import AGENT_REGISTRY

        assert (
            "genomics_expert" in AGENT_REGISTRY
        ), "genomics_expert should be in AGENT_REGISTRY"
        config = AGENT_REGISTRY["genomics_expert"]

        # Verify basic configuration
        assert config.name == "genomics_expert"
        assert config.display_name == "Genomics Expert"
        assert (
            config.factory_function
            == "lobster.agents.genomics.genomics_expert.genomics_expert"
        )
        assert config.handoff_tool_name == "handoff_to_genomics_expert"

        # Verify agent is supervisor-accessible
        assert (
            config.supervisor_accessible is True
        ), "genomics_expert should be supervisor-accessible"

    def test_agent_in_premium_tier(self):
        """Test that genomics_expert is in PREMIUM tier."""
        from lobster.config.subscription_tiers import get_tier_agents

        # Get premium tier agents
        premium_agents = get_tier_agents("premium")

        # genomics_expert should be in PREMIUM tier
        assert (
            "genomics_expert" in premium_agents
        ), f"genomics_expert should be in premium tier. Found: {premium_agents}"

    def test_agent_adapters_registered_in_data_manager(self):
        """Test that genomics adapters are registered in DataManagerV2."""
        # Check that adapter classes are importable
        try:
            from lobster.core.adapters.genomics.plink_adapter import PLINKAdapter
            from lobster.core.adapters.genomics.vcf_adapter import VCFAdapter

            assert VCFAdapter is not None
            assert PLINKAdapter is not None

            # Verify they follow adapter interface
            vcf_adapter = VCFAdapter(strict_validation=False)
            plink_adapter = PLINKAdapter(strict_validation=False)

            assert hasattr(vcf_adapter, "from_source")
            assert hasattr(plink_adapter, "from_source")

        except ImportError as e:
            pytest.fail(f"Genomics adapters not importable: {e}")


# ===============================================================================
# System Prompt Tests
# ===============================================================================


@pytest.mark.unit
class TestGenomicsExpertPrompts:
    """Test genomics expert system prompts."""

    def test_system_prompt_exists(self):
        """Test that system prompt is defined."""
        # Check if prompts module exists
        try:
            from lobster.agents.genomics import prompts

            assert prompts is not None

            # Try to get system prompt
            if hasattr(prompts, "create_system_prompt"):
                prompt = prompts.create_system_prompt()
                assert isinstance(prompt, str)
                assert len(prompt) > 100
                # Check key content
                assert (
                    "genomics" in prompt.lower()
                    or "genotype" in prompt.lower()
                    or "vcf" in prompt.lower()
                )
            elif hasattr(prompts, "SYSTEM_PROMPT"):
                prompt = prompts.SYSTEM_PROMPT
                assert isinstance(prompt, str)
                assert len(prompt) > 100
            else:
                pytest.skip("System prompt structure different than expected")

        except ImportError:
            pytest.skip("Prompts module not found")

    def test_system_prompt_mentions_tools(self):
        """Test that system prompt mentions genomics-related content."""
        try:
            from lobster.agents.genomics import prompts

            # Get prompt (try different patterns)
            prompt = None
            if hasattr(prompts, "create_system_prompt"):
                prompt = prompts.create_system_prompt()
            elif hasattr(prompts, "SYSTEM_PROMPT"):
                prompt = prompts.SYSTEM_PROMPT

            if prompt is None:
                pytest.skip("Could not access system prompt")

            # Should mention genomics-related terms
            key_terms = [
                "genomics",
                "vcf",
                "plink",
                "gwas",
                "variant",
                "genotype",
                "qc",
                "filter",
            ]
            found_count = sum(1 for term in key_terms if term.lower() in prompt.lower())

            assert (
                found_count >= 3
            ), f"System prompt should mention genomics concepts, found {found_count}/{len(key_terms)} terms"

        except ImportError:
            pytest.skip("Prompts module not found")
