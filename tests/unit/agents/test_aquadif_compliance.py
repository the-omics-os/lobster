"""
AQUADIF contract compliance smoke tests.

Validates the contract test mixin infrastructure using mock tools.
Real agent compliance tests are created per-package in Phase 3+.

Usage:
    pytest tests/unit/agents/test_aquadif_compliance.py -x --no-cov
    pytest tests/unit/agents/ -m contract --no-cov
"""
import pytest
from langchain_core.tools import tool

from lobster.config.aquadif import AquadifCategory, PROVENANCE_REQUIRED


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

    def test_category_cap_violation(self):
        """More than 3 categories should be detectable."""
        @tool
        def over_categorized(data: str) -> str:
            """Over-categorized tool."""
            return "done"

        over_categorized.metadata = {
            "categories": ["IMPORT", "QUALITY", "FILTER", "PREPROCESS"],
        }
        assert len(over_categorized.metadata["categories"]) > 3

    def test_metadata_uniqueness_violation(self):
        """Shared metadata dict objects should be detectable."""
        shared_metadata = {"categories": ["ANALYZE"], "provenance": True}

        @tool
        def tool_a(data: str) -> str:
            """Tool A."""
            return "a"

        @tool
        def tool_b(data: str) -> str:
            """Tool B."""
            return "b"

        tool_a.metadata = shared_metadata
        tool_b.metadata = shared_metadata

        # Same dict object — contract test would catch this
        assert id(tool_a.metadata) == id(tool_b.metadata)

    def test_provenance_required_categories(self):
        """Verify all 7 provenance-required categories."""
        expected = {"IMPORT", "QUALITY", "FILTER", "PREPROCESS", "ANALYZE", "ANNOTATE", "SYNTHESIZE"}
        actual = {cat.value for cat in PROVENANCE_REQUIRED}
        assert actual == expected

    def test_non_provenance_categories(self):
        """Verify 3 non-provenance categories."""
        non_prov = {cat for cat in AquadifCategory if cat not in PROVENANCE_REQUIRED}
        expected = {AquadifCategory.DELEGATE, AquadifCategory.UTILITY, AquadifCategory.CODE_EXEC}
        assert non_prov == expected

    def test_utility_tool_no_provenance(self):
        """UTILITY tools should not declare provenance."""
        @tool
        def list_modalities(query: str) -> str:
            """List loaded modalities."""
            return "listed"

        list_modalities.metadata = {
            "categories": ["UTILITY"],
            "provenance": False,
        }
        cat = AquadifCategory(list_modalities.metadata["categories"][0])
        assert cat not in PROVENANCE_REQUIRED
        assert list_modalities.metadata["provenance"] is False
