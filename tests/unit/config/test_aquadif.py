"""
Unit tests for AQUADIF category enum and provenance configuration.

These tests validate the AQUADIF taxonomy structure that all contract tests
depend on. They ensure the enum has exactly 10 categories with correct names,
values, and provenance requirements.
"""

import pytest

from lobster.config.aquadif import (
    AquadifCategory,
    PROVENANCE_REQUIRED,
    has_provenance_call,
    requires_provenance,
)


class TestAquadifCategory:
    """Test AQUADIF category enum structure and behavior."""

    def test_has_exactly_ten_categories(self):
        """Verify the enum has exactly 10 category members."""
        assert len(AquadifCategory) == 10, (
            f"AQUADIF must have exactly 10 categories, found {len(AquadifCategory)}"
        )

    def test_category_names(self):
        """Verify all 10 expected category names are present."""
        expected_names = {
            "IMPORT",
            "QUALITY",
            "FILTER",
            "PREPROCESS",
            "ANALYZE",
            "ANNOTATE",
            "DELEGATE",
            "SYNTHESIZE",
            "UTILITY",
            "CODE_EXEC",
        }
        actual_names = {cat.name for cat in AquadifCategory}

        assert actual_names == expected_names, (
            f"Category names mismatch.\n"
            f"Expected: {sorted(expected_names)}\n"
            f"Actual: {sorted(actual_names)}\n"
            f"Missing: {sorted(expected_names - actual_names)}\n"
            f"Extra: {sorted(actual_names - expected_names)}"
        )

    def test_category_values_match_names(self):
        """Verify each category's value equals its name (e.g., IMPORT.value == 'IMPORT')."""
        for category in AquadifCategory:
            assert category.value == category.name, (
                f"Category {category.name} has mismatched value '{category.value}'. "
                f"Values should match names for consistency."
            )

    def test_provenance_required_has_seven_members(self):
        """Verify PROVENANCE_REQUIRED has exactly 7 category members."""
        assert len(PROVENANCE_REQUIRED) == 7, (
            f"PROVENANCE_REQUIRED must have exactly 7 categories, found {len(PROVENANCE_REQUIRED)}"
        )

    def test_provenance_required_categories(self):
        """Verify exact membership of PROVENANCE_REQUIRED set."""
        expected = {
            AquadifCategory.IMPORT,
            AquadifCategory.QUALITY,
            AquadifCategory.FILTER,
            AquadifCategory.PREPROCESS,
            AquadifCategory.ANALYZE,
            AquadifCategory.ANNOTATE,
            AquadifCategory.SYNTHESIZE,
        }
        assert PROVENANCE_REQUIRED == expected, (
            f"PROVENANCE_REQUIRED membership mismatch.\n"
            f"Expected: {sorted(c.value for c in expected)}\n"
            f"Actual: {sorted(c.value for c in PROVENANCE_REQUIRED)}\n"
            f"Missing: {sorted(c.value for c in (expected - PROVENANCE_REQUIRED))}\n"
            f"Extra: {sorted(c.value for c in (PROVENANCE_REQUIRED - expected))}"
        )

    def test_non_provenance_categories(self):
        """Verify DELEGATE, UTILITY, CODE_EXEC are NOT in PROVENANCE_REQUIRED."""
        non_provenance = {
            AquadifCategory.DELEGATE,
            AquadifCategory.UTILITY,
            AquadifCategory.CODE_EXEC,
        }

        for category in non_provenance:
            assert category not in PROVENANCE_REQUIRED, (
                f"{category.value} should NOT require provenance but is in PROVENANCE_REQUIRED"
            )

    def test_requires_provenance_true(self):
        """Verify requires_provenance() returns True for IMPORT category."""
        assert requires_provenance("IMPORT") is True, (
            "IMPORT category should require provenance"
        )

    def test_requires_provenance_false(self):
        """Verify requires_provenance() returns False for UTILITY category."""
        assert requires_provenance("UTILITY") is False, (
            "UTILITY category should NOT require provenance"
        )

    def test_requires_provenance_invalid(self):
        """Verify requires_provenance() raises ValueError for invalid category string."""
        with pytest.raises(ValueError) as exc_info:
            requires_provenance("INVALID")

        error_message = str(exc_info.value)
        assert "INVALID" in error_message, "Error should mention the invalid category"
        assert "not a valid AquadifCategory" in error_message, (
            "Error should explain it's not a valid category"
        )

    def test_string_comparison(self):
        """Verify string enum behavior: AquadifCategory.IMPORT == 'IMPORT' is True."""
        # This validates the (str, Enum) base class works correctly
        assert AquadifCategory.IMPORT == "IMPORT", (
            "String enum should allow direct string comparison"
        )
        assert AquadifCategory.QUALITY == "QUALITY"
        assert AquadifCategory.UTILITY == "UTILITY"

        # Also verify value access
        assert AquadifCategory.IMPORT.value == "IMPORT"


class TestHasProvenanceCall:
    """Test has_provenance_call AST helper function."""

    def test_detects_ir_keyword(self):
        """Should detect log_tool_usage with ir= keyword argument."""
        import ast

        tree = ast.parse('data_manager.log_tool_usage("test", params, stats, ir=ir)')
        assert has_provenance_call(tree) is True

    def test_rejects_without_ir(self):
        """Should return False when log_tool_usage has no ir= keyword."""
        import ast

        tree = ast.parse('data_manager.log_tool_usage("test", params, stats)')
        assert has_provenance_call(tree) is False

    def test_detects_ir_in_multiline(self):
        """Should detect ir= in multi-line function body."""
        import ast

        source = """
def my_tool(data):
    result, stats, ir = service.analyze(adata)
    data_manager.log_tool_usage("my_tool", {"data": data}, stats, ir=ir)
    return "done"
"""
        tree = ast.parse(source)
        assert has_provenance_call(tree) is True

    def test_rejects_unrelated_call(self):
        """Should not match unrelated method names."""
        import ast

        tree = ast.parse('data_manager.save_results("test", ir=ir)')
        assert has_provenance_call(tree) is False

    def test_empty_tree(self):
        """Should return False for empty/trivial AST."""
        import ast

        tree = ast.parse("pass")
        assert has_provenance_call(tree) is False
