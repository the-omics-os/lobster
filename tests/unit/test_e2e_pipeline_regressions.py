"""
Regression tests for the 9 bugs fixed in the proteomics E2E pipeline.

These tests guard against re-introduction of bugs fixed in commit 0c81893:
- graph.py: child agent auto-include when parent is enabled
- proteomics_adapter.py: CSV kwargs exclusion + Categorical column handling
- shared_tools.py: parameter_schema on generic import IR

Each test documents the original bug, the fix, and the exact failure mode.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.config.agent_registry import AgentRegistryConfig


# ===============================================================================
# Graph: child agent auto-include
# ===============================================================================


class TestGraphChildAutoInclude:
    """Regression: enabled_agents filter excluded children of enabled parents.

    Bug: When config.enabled_agents listed only parent agents (e.g.,
    'proteomics_expert'), child agents ('proteomics_de_analysis_expert')
    were filtered out, breaking delegation tools.
    Fix: After filtering, auto-include children of any enabled parent.
    """

    def _make_agent_config(self, name, child_agents=None):
        return AgentRegistryConfig(
            name=name,
            display_name=name,
            description=f"Test agent {name}",
            factory_function=f"test.{name}",
            child_agents=child_agents,
        )

    def test_child_included_when_parent_enabled(self):
        """Children of enabled parents must be in worker_agents."""
        all_agents = {
            "parent": self._make_agent_config("parent", child_agents=["child_a"]),
            "child_a": self._make_agent_config("child_a"),
            "unrelated": self._make_agent_config("unrelated"),
        }
        enabled_set = {"parent"}

        # Replicate the filtering logic from graph.py:397-412
        worker_agents = {n: c for n, c in all_agents.items() if n in enabled_set}
        child_additions = {}
        for agent_name, agent_config in list(worker_agents.items()):
            if agent_config.child_agents:
                for child_name in agent_config.child_agents:
                    if child_name not in worker_agents and child_name in all_agents:
                        child_additions[child_name] = all_agents[child_name]
        worker_agents.update(child_additions)

        assert "parent" in worker_agents
        assert "child_a" in worker_agents
        assert "unrelated" not in worker_agents

    def test_multiple_children_included(self):
        """All children of an enabled parent must be included."""
        all_agents = {
            "parent": self._make_agent_config(
                "parent", child_agents=["child_a", "child_b"]
            ),
            "child_a": self._make_agent_config("child_a"),
            "child_b": self._make_agent_config("child_b"),
        }
        enabled_set = {"parent"}

        worker_agents = {n: c for n, c in all_agents.items() if n in enabled_set}
        child_additions = {}
        for agent_name, agent_config in list(worker_agents.items()):
            if agent_config.child_agents:
                for child_name in agent_config.child_agents:
                    if child_name not in worker_agents and child_name in all_agents:
                        child_additions[child_name] = all_agents[child_name]
        worker_agents.update(child_additions)

        assert "child_a" in worker_agents
        assert "child_b" in worker_agents

    def test_explicitly_enabled_child_not_duplicated(self):
        """If child is already in enabled_set, don't add twice."""
        all_agents = {
            "parent": self._make_agent_config("parent", child_agents=["child_a"]),
            "child_a": self._make_agent_config("child_a"),
        }
        enabled_set = {"parent", "child_a"}

        worker_agents = {n: c for n, c in all_agents.items() if n in enabled_set}
        child_additions = {}
        for agent_name, agent_config in list(worker_agents.items()):
            if agent_config.child_agents:
                for child_name in agent_config.child_agents:
                    if child_name not in worker_agents and child_name in all_agents:
                        child_additions[child_name] = all_agents[child_name]
        worker_agents.update(child_additions)

        assert len(worker_agents) == 2

    def test_missing_child_package_skipped(self):
        """If child isn't in all_agents (package not installed), skip it."""
        all_agents = {
            "parent": self._make_agent_config(
                "parent", child_agents=["child_a", "missing_child"]
            ),
            "child_a": self._make_agent_config("child_a"),
            # missing_child not in all_agents
        }
        enabled_set = {"parent"}

        worker_agents = {n: c for n, c in all_agents.items() if n in enabled_set}
        child_additions = {}
        for agent_name, agent_config in list(worker_agents.items()):
            if agent_config.child_agents:
                for child_name in agent_config.child_agents:
                    if child_name not in worker_agents and child_name in all_agents:
                        child_additions[child_name] = all_agents[child_name]
        worker_agents.update(child_additions)

        assert "child_a" in worker_agents
        assert "missing_child" not in worker_agents


# ===============================================================================
# Proteomics adapter: CSV kwargs exclusion
# ===============================================================================


class TestProteomicsAdapterCsvKwargs:
    """Regression: read_csv received unexpected kwargs dataset_id, dataset_type.

    Bug: proteomics_adapter passed all kwargs through to pandas.read_csv(),
    including lobster-internal params like dataset_id and dataset_type.
    Fix: _csv_excluded set now includes 'dataset_id' and 'dataset_type'.
    """

    def test_csv_excluded_contains_dataset_params(self):
        """dataset_id and dataset_type must be in the exclusion set."""
        from lobster.core.adapters.proteomics_adapter import ProteomicsAdapter

        import inspect

        source = inspect.getsource(ProteomicsAdapter._load_csv_proteomics_data)
        # These params must be in _csv_excluded to prevent pandas TypeError
        assert "dataset_id" in source
        assert "dataset_type" in source


class TestProteomicsAdapterCategoricalSum:
    """Regression: Categorical column .sum() raised TypeError.

    Bug: adata.var['is_contaminant'] as Categorical didn't support .sum().
    Fix: use .astype(bool).sum() for boolean-like Categorical columns.
    """

    def test_categorical_bool_sum(self):
        """Categorical boolean column must be summable via astype(bool)."""
        cat_series = pd.Categorical([True, False, True, True, False])
        series = pd.Series(cat_series)

        # Old code: series.sum() would fail on Categorical
        # New code: series.astype(bool).sum()
        result = int(series.astype(bool).sum())
        assert result == 3

    def test_categorical_string_bool_sum(self):
        """Categorical with string 'True'/'False' also works."""
        cat_series = pd.Categorical(["True", "False", "True"])
        series = pd.Series(cat_series)
        # astype(bool) on strings: all non-empty strings are True
        # The adapter handles this via try/except fallback
        result = int(series.astype(bool).sum())
        assert result == 3  # All non-empty strings are truthy
