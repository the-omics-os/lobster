"""
AQUADIF contract compliance enforcement tests.

Validates the contract test mixin catches violations correctly using mock tools.
Real agent compliance tests are created per-package in Phase 3+.

Usage:
    pytest tests/unit/agents/test_aquadif_compliance.py -x --no-cov
    pytest tests/unit/agents/ -m contract --no-cov
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import tool

from lobster.config.aquadif import PROVENANCE_REQUIRED, AquadifCategory
from lobster.testing.contract_mixins import AgentContractTestMixin


def _make_tool_with_metadata(name, categories, provenance=False):
    """Helper: create a mock tool with AQUADIF metadata."""

    @tool
    def dummy_tool(data: str) -> str:
        """Dummy tool for testing."""
        return "done"

    dummy_tool.name = name
    dummy_tool.metadata = {
        "categories": categories,
        "provenance": provenance,
    }
    return dummy_tool


class _MockMixin(AgentContractTestMixin):
    """Test mixin subclass that returns controlled tools instead of calling a factory."""

    agent_module = "test.mock"
    factory_name = "mock_factory"
    tools_required = True
    _mock_tools = []
    _tools_cache = {}  # Fresh cache per test

    def _get_tools_from_factory(self):
        return self._mock_tools


@pytest.mark.contract
class TestAquadifContractSmoke:
    """Smoke tests for AQUADIF contract test infrastructure."""

    def test_valid_metadata_structure(self):
        """A tool with correct metadata should pass all basic checks."""

        @tool
        def import_data(path: str) -> str:
            """Import data from file."""
            return "imported"

        import_data.metadata = {
            "categories": ["IMPORT"],
            "provenance": True,
        }
        import_data.tags = ["IMPORT"]

        assert "categories" in import_data.metadata
        assert isinstance(import_data.metadata["categories"], list)
        assert len(import_data.metadata["categories"]) <= 3
        cat = AquadifCategory(import_data.metadata["categories"][0])
        assert cat in PROVENANCE_REQUIRED
        assert import_data.metadata["provenance"] is True

    def test_invalid_category_detected(self):
        """Invalid category should raise ValueError."""
        with pytest.raises(ValueError):
            AquadifCategory("INVALID_CATEGORY")

    def test_provenance_required_categories(self):
        """Verify all 7 provenance-required categories."""
        expected = {
            "IMPORT",
            "QUALITY",
            "FILTER",
            "PREPROCESS",
            "ANALYZE",
            "ANNOTATE",
            "SYNTHESIZE",
        }
        actual = {cat.value for cat in PROVENANCE_REQUIRED}
        assert actual == expected

    def test_non_provenance_categories(self):
        """Verify 3 non-provenance categories."""
        non_prov = {cat for cat in AquadifCategory if cat not in PROVENANCE_REQUIRED}
        expected = {
            AquadifCategory.DELEGATE,
            AquadifCategory.UTILITY,
            AquadifCategory.CODE_EXEC,
        }
        assert non_prov == expected


@pytest.mark.contract
class TestMixinEnforcementCatchesViolations:
    """Verify the mixin's test methods actually catch violations (not tautologies)."""

    def test_category_cap_violation_caught(self):
        """Mixin catches tools with more than 3 categories."""
        mixin = _MockMixin()
        mixin._mock_tools = [
            _make_tool_with_metadata(
                "over_categorized", ["IMPORT", "QUALITY", "FILTER", "PREPROCESS"]
            ),
        ]

        with pytest.raises(AssertionError, match="exceed 3-category limit"):
            mixin.test_categories_capped_at_three()

    def test_metadata_uniqueness_violation_caught(self):
        """Mixin catches tools sharing the same metadata dict object."""
        shared_metadata = {"categories": ["ANALYZE"], "provenance": True}

        tool_a = _make_tool_with_metadata("tool_a", ["ANALYZE"], True)
        tool_b = _make_tool_with_metadata("tool_b", ["ANALYZE"], True)
        # Force shared reference
        tool_a.metadata = shared_metadata
        tool_b.metadata = shared_metadata

        mixin = _MockMixin()
        mixin._mock_tools = [tool_a, tool_b]

        with pytest.raises(AssertionError, match="share metadata dict objects"):
            mixin.test_metadata_objects_are_unique()

    def test_provenance_ordering_violation_caught(self):
        """Mixin catches provenance-required category buried behind non-provenance primary."""
        mixin = _MockMixin()
        mixin._mock_tools = [
            _make_tool_with_metadata("sneaky_tool", ["UTILITY", "IMPORT"], False),
        ]

        with pytest.raises(
            AssertionError, match="buried behind non-provenance primary"
        ):
            mixin.test_provenance_categories_not_buried()

    def test_missing_provenance_flag_caught(self):
        """Mixin catches tools with provenance-required primary but provenance=False."""
        mixin = _MockMixin()
        mixin._mock_tools = [
            _make_tool_with_metadata("bad_import", ["IMPORT"], provenance=False),
        ]

        with pytest.raises(AssertionError, match="requires provenance=True"):
            mixin.test_provenance_tools_have_flag()

    def test_factory_failure_is_not_skipped(self):
        """Factory RuntimeError should FAIL the test, not skip it."""

        class _BrokenMixin(AgentContractTestMixin):
            agent_module = "test.broken"
            factory_name = "broken_factory"
            _tools_cache = {}

            def _get_factory(self):
                raise RuntimeError("Factory is broken!")

        mixin = _BrokenMixin()

        with pytest.raises(RuntimeError, match="Factory is broken"):
            mixin._get_tools_from_factory()

    def test_empty_tools_fails_when_required(self):
        """Empty tools list should FAIL when tools_required=True."""
        mixin = _MockMixin()
        mixin._mock_tools = []

        with pytest.raises(AssertionError, match="returned no tools"):
            mixin._require_tools()

    def test_empty_tools_skips_when_not_required(self):
        """Empty tools list should skip when tools_required=False."""
        mixin = _MockMixin()
        mixin._mock_tools = []
        mixin.tools_required = False

        with pytest.raises(pytest.skip.Exception):
            mixin._require_tools()
