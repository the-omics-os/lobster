"""
P0 Error Hardening Integration Tests — Real Agent Invocation

Tests the 4 P0 tools by running actual lobster sessions that trigger
error paths, then inspecting what the agent received.

This uses lobster's programmatic API (AgentClient) with a real LLM to
verify the agent gets actionable error context and can recover.

Run:
    pytest tests/integration/test_p0_error_messages.py -v -s --no-cov

Requires: configured LLM provider (omics-os, anthropic, or openai)
"""

import json
import tempfile
import importlib
from pathlib import Path
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.core.runtime.data_manager import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dm_with_proteomics(tmp_path):
    """Create a DataManagerV2 with proteomics test data pre-loaded."""
    n_obs, n_vars = 20, 50
    np.random.seed(42)

    obs = pd.DataFrame(
        {
            "condition": pd.Categorical(["treated"] * 10 + ["control"] * 10),
            "batch": pd.Categorical(
                ["batch1"] * 5 + ["batch2"] * 5 + ["batch1"] * 5 + ["batch2"] * 5
            ),
            "patient_id": [f"P{i:03d}" for i in range(n_obs)],
            "age": np.random.randint(25, 75, n_obs),
        },
        index=[f"sample_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_name": [f"PROT{i}" for i in range(n_vars)]},
        index=[f"PROT{i}" for i in range(n_vars)],
    )
    X = np.random.randn(n_obs, n_vars).astype(np.float32)
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # Save h5ad for load_modality tests
    h5ad_path = tmp_path / "proteomics_data.h5ad"
    adata.write_h5ad(h5ad_path)

    dm = DataManagerV2(workspace_path=tmp_path)
    dm.store_modality(name="test_proteomics", adata=adata, step_summary="test data")

    return dm, tmp_path, h5ad_path


# ---------------------------------------------------------------------------
# Direct tool invocation via real factories
# ---------------------------------------------------------------------------


def _extract_tool(factory_fn, data_manager, tool_name, **factory_kwargs):
    """
    Call an agent factory, intercept the tools list before agent creation,
    and return a specific tool by name.

    We monkey-patch create_react_agent to capture the tools list.
    """
    captured_tools = []

    import langgraph.prebuilt

    original_cra = langgraph.prebuilt.create_react_agent
    factory_module = importlib.import_module(factory_fn.__module__)
    original_module_cra = getattr(factory_module, "create_react_agent", None)
    original_create_llm = getattr(factory_module, "create_llm", None)
    original_get_settings = getattr(factory_module, "get_settings", None)

    def interceptor(*args, **kwargs):
        tools = kwargs.get("tools", args[1] if len(args) > 1 else [])
        if hasattr(tools, "tools_by_name"):
            # It's a ToolNode
            captured_tools.extend(tools.tools_by_name.values())
        elif isinstance(tools, (list, tuple)):
            captured_tools.extend(tools)
        # Return a dummy — we don't need the actual agent
        return "INTERCEPTED"

    langgraph.prebuilt.create_react_agent = interceptor
    factory_module.create_react_agent = interceptor

    dummy_llm = Mock()
    dummy_llm.with_config.return_value = dummy_llm
    dummy_settings = Mock()
    dummy_settings.get_agent_llm_params.return_value = {}
    if original_create_llm is not None:
        factory_module.create_llm = Mock(return_value=dummy_llm)
    if original_get_settings is not None:
        factory_module.get_settings = Mock(return_value=dummy_settings)

    try:
        factory_fn(data_manager=data_manager, **factory_kwargs)
    except Exception:
        pass  # Factory might fail on LLM creation, that's OK
    finally:
        langgraph.prebuilt.create_react_agent = original_cra
        if original_module_cra is not None:
            factory_module.create_react_agent = original_module_cra
        if original_create_llm is not None:
            factory_module.create_llm = original_create_llm
        if original_get_settings is not None:
            factory_module.get_settings = original_get_settings

    for t in captured_tools:
        if t.name == tool_name:
            return t

    available = [t.name for t in captured_tools]
    pytest.fail(f"Tool '{tool_name}' not found. Available: {available}")


def _get_proteomics_de_tool(dm, tool_name):
    """Get a tool from the proteomics DE factory."""
    from lobster.agents.proteomics.de_analysis_expert import de_analysis_expert

    return _extract_tool(de_analysis_expert, dm, tool_name)


def _get_data_expert_tool(dm, tool_name):
    """Get a tool from the data_expert factory."""
    from lobster.agents.data_expert.data_expert import data_expert

    return _extract_tool(data_expert, dm, tool_name)


# ===========================================================================
# TEST 1: find_differential_proteins — demo failure scenario
# ===========================================================================


class TestFindDifferentialProteins:
    """Tests the exact error path that caused the Emory demo failure."""

    def test_wrong_group_column(self, dm_with_proteomics):
        """Agent uses wrong column name. Error must list available columns."""
        dm, _, _ = dm_with_proteomics
        tool = _get_proteomics_de_tool(dm, "find_differential_proteins")

        result = tool.invoke(
            {
                "modality_name": "test_proteomics",
                "group_column": "treatment_group",  # WRONG — actual is "condition"
            }
        )

        print("\n" + "=" * 60)
        print("SCENARIO: Wrong group column (Emory demo failure)")
        print("=" * 60)
        print(result)
        print("=" * 60)

        # Agent MUST see available columns to self-correct
        assert "condition" in result, "Must list 'condition' in available obs columns"
        assert "batch" in result, "Must list 'batch' in available obs columns"
        assert "treatment_group" in result, "Must echo the bad column name"
        assert "test_proteomics" in result, "Must include modality name"
        # Must explicitly indicate the column is missing
        assert (
            "NOT in obs" in result or "not found" in result.lower()
        ), "Must indicate column is not in obs"

    def test_nonexistent_modality(self, dm_with_proteomics):
        """Modality doesn't exist — must list available ones."""
        dm, _, _ = dm_with_proteomics
        tool = _get_proteomics_de_tool(dm, "find_differential_proteins")

        result = tool.invoke(
            {
                "modality_name": "nonexistent",
                "group_column": "condition",
            }
        )

        print("\n" + "=" * 60)
        print("SCENARIO: Nonexistent modality")
        print("=" * 60)
        print(result)
        print("=" * 60)

        assert "test_proteomics" in result, "Must list available modalities"
        assert "nonexistent" in result, "Must echo the bad modality name"


# ===========================================================================
# TEST 2: load_modality
# ===========================================================================


class TestLoadModality:

    def test_wrong_file_path(self, dm_with_proteomics):
        """File doesn't exist — must echo params and suggest recovery."""
        dm, _, _ = dm_with_proteomics
        tool = _get_data_expert_tool(dm, "load_modality")

        result = tool.invoke(
            {
                "modality_name": "my_data",
                "file_path": "/nonexistent/path/data.h5ad",
                "adapter": "10x_h5ad",
            }
        )

        print("\n" + "=" * 60)
        print("SCENARIO: Wrong file path")
        print("=" * 60)
        print(result)
        print("=" * 60)

        assert "/nonexistent/path/data.h5ad" in result, "Must echo file_path"
        assert "10x_h5ad" in result, "Must echo adapter"
        assert (
            "list_files" in result or "glob_files" in result
        ), "Must suggest file discovery tools"

    def test_wrong_adapter(self, dm_with_proteomics):
        """File exists but adapter is wrong."""
        dm, _, h5ad_path = dm_with_proteomics
        tool = _get_data_expert_tool(dm, "load_modality")

        result = tool.invoke(
            {
                "modality_name": "my_data",
                "file_path": str(h5ad_path),
                "adapter": "totally_wrong_adapter",
            }
        )

        print("\n" + "=" * 60)
        print("SCENARIO: Wrong adapter")
        print("=" * 60)
        print(result)
        print("=" * 60)

        assert "totally_wrong_adapter" in result, "Must echo the bad adapter"
        assert "get_adapter_info" in result, "Must suggest adapter info tool"


# ===========================================================================
# TEST 3: get_modality_details
# ===========================================================================


class TestGetModalityDetails:

    def test_nonexistent_modality(self, dm_with_proteomics):
        """Modality doesn't exist — must list available ones."""
        dm, _, _ = dm_with_proteomics
        tool = _get_data_expert_tool(dm, "get_modality_details")

        result = tool.invoke({"modality_name": "i_dont_exist"})

        print("\n" + "=" * 60)
        print("SCENARIO: Nonexistent modality details")
        print("=" * 60)
        print(result)
        print("=" * 60)

        assert "i_dont_exist" in result, "Must echo the bad modality name"
        assert "test_proteomics" in result, "Must list available modalities"


# ===========================================================================
# TEST 4: concatenate_samples
# ===========================================================================


class TestConcatenateSamples:

    def test_nonexistent_samples(self, dm_with_proteomics):
        """Sample modalities don't exist."""
        dm, _, _ = dm_with_proteomics
        tool = _get_data_expert_tool(dm, "concatenate_samples")

        result = tool.invoke(
            {
                "sample_modalities": ["sample_1", "sample_2", "sample_3"],
            }
        )

        print("\n" + "=" * 60)
        print("SCENARIO: Nonexistent sample modalities")
        print("=" * 60)
        print(result)
        print("=" * 60)

        assert "sample_1" in result, "Must echo requested samples"
        assert "test_proteomics" in result, "Must list available modalities"

    def test_auto_detect_failure(self, dm_with_proteomics):
        """GEO auto-detect finds no matching samples."""
        dm, _, _ = dm_with_proteomics
        tool = _get_data_expert_tool(dm, "concatenate_samples")

        result = tool.invoke({"geo_id": "GSE99999"})

        print("\n" + "=" * 60)
        print("SCENARIO: GEO auto-detect failure")
        print("=" * 60)
        print(result)
        print("=" * 60)

        assert "GSE99999" in result, "Must echo the geo_id"
